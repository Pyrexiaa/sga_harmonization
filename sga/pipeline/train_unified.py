"""Unified (pooled, cross-domain harmonized) training for every classifier family."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from torch.optim.lr_scheduler import CosineAnnealingLR

from sga.config import (
    CONSTANT_ZERO_FEATURES,
    INDIA_BINARY_FEATURES,
    INDIA_REGRESSION_FEATURES,
    LABEL,
    MALAYSIA_MULTICLASS_FEATURES,
    MALAYSIA_REGRESSION_FEATURES,
    N_FOLDS_CV,
    PREV_PREGNANCY_FEATURES,
    SEED,
)
from sga.data.cleaning import remove_illogical_values
from sga.data.scaling import descale_feature
from sga.evaluation.country import append_fold_rows, evaluate_splits, save_country_results
from sga.imputation.apply import impute_df, select_features_with_threshold
from sga.imputation.fold_imputers import fit_fold_imputers
from sga.models.architecture import MODEL_SIZES
from sga.models.ensemble import build_stacking_classifier
from sga.models.hyperparameters import (
    catboost_hyperparameters,
    lr_hyperparameters,
    rf_hyperparameters,
    svc_hyperparameters,
)
from sga.models.loops import test_dnn, train_dnn
from sga.models.torch_utils import (
    DEVICE,
    compute_permutation_importance,
    convert_to_tensor_dataloader,
)
from sga.pipeline.dataset import (
    cast_common_types,
    process_raw_train_and_test_df,
    scale_sample_train_and_test_df,
    scale_training_partition,
    separate_df_and_df_add_on,
)
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA
from sga.reporting.artifacts import save_output_layer_dict, save_train_test_set
from sga.reporting.importance import (
    compute_and_plot_permutation_importance,
    save_permutation_importance_list,
)
from sga.reporting.metrics_tables import (
    calc_metrics_with_ci,
    calculate_metrics,
    display_metrics_updated,
    display_ml_metrics,
)
from sga.reporting.plots import (
    display_cm,
    display_roc_curve_binary,
    display_training_loss,
    display_validation_loss,
)
from sga.reporting.shap_analysis import display_shap, new_shap_dict, perform_SHAP

# Metric callables evaluated for every fold, in the order the callers unpack.
METRICS_LIST = [
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
]

# Columns used internally to carry the cohort of origin and the pregnancy
# identifier through the row-dropping physiological filter, so ``country_arr`` and
# ``cluster_ids`` can never fall out of alignment with the surviving test rows.
_COUNTRY_COLUMN = "__country__"
_CLUSTER_COLUMN = "__cluster__"

SKLEARN_MODEL_TYPES = ("lr", "rf", "svc", "stacking")


# ── Feature selection ────────────────────────────────────────────────────────


@dataclass
class CrossDomainFeatures:
    """Cross-domain candidate features split into retained and removed sets."""

    msia_multiclass: list = field(default_factory=list)
    msia_regression: list = field(default_factory=list)
    india_binary: list = field(default_factory=list)
    india_regression: list = field(default_factory=list)
    removed_msia_multiclass: list = field(default_factory=list)
    removed_msia_regression: list = field(default_factory=list)
    removed_india_binary: list = field(default_factory=list)
    removed_india_regression: list = field(default_factory=list)

    @property
    def msia_impute(self):
        """Malaysia-native features imputed INTO the India cohort."""
        return self.msia_multiclass + self.msia_regression

    @property
    def india_impute(self):
        """India-native features imputed INTO the Malaysia cohort."""
        return self.india_binary + self.india_regression

    @property
    def removed_from_malaysia(self):
        """Malaysia-native features dropped from both cohorts."""
        return self.removed_msia_multiclass + self.removed_msia_regression

    @property
    def removed_from_india(self):
        """India-native features dropped from both cohorts."""
        return self.removed_india_binary + self.removed_india_regression

    @property
    def imputation_targets(self):
        """``(feature, kind)`` pairs for the retained features, for refitting."""
        pairs = (
            [(f, "multiclass") for f in self.msia_multiclass]
            + [(f, "regression") for f in self.msia_regression]
            + [(f, "binary") for f in self.india_binary]
            + [(f, "regression") for f in self.india_regression]
        )
        return [(f, kind) for f, kind in pairs if f not in CONSTANT_ZERO_FEATURES]


def select_cross_domain_features(accuracy_threshold, drop_prev_pregnancy=False,
                                 verbose=False, selected_features=None):
    """Gate the cross-domain candidates on their imputation quality."""
    if selected_features is not None:
        keep = set(selected_features)

        def split(candidates):
            return (
                [f for f in candidates if f in keep],
                [f for f in candidates if f not in keep],
            )

        msia_multiclass, removed_msia_multiclass = split(MALAYSIA_MULTICLASS_FEATURES)
        msia_regression, removed_msia_regression = split(MALAYSIA_REGRESSION_FEATURES)
        india_binary, removed_india_binary = split(
            [
                feature
                for feature in INDIA_BINARY_FEATURES
                if not (drop_prev_pregnancy and feature in PREV_PREGNANCY_FEATURES)
            ]
        )
        india_regression, removed_india_regression = split(INDIA_REGRESSION_FEATURES)
        return CrossDomainFeatures(
            msia_multiclass=msia_multiclass,
            msia_regression=msia_regression,
            india_binary=india_binary,
            india_regression=india_regression,
            removed_msia_multiclass=removed_msia_multiclass,
            removed_msia_regression=removed_msia_regression,
            removed_india_binary=removed_india_binary,
            removed_india_regression=removed_india_regression,
        )

    msia_multiclass, removed_msia_multiclass = select_features_with_threshold(
        MALAYSIA_MULTICLASS_FEATURES, accuracy_threshold, "multiclass", verbose=verbose
    )
    msia_regression, removed_msia_regression = select_features_with_threshold(
        MALAYSIA_REGRESSION_FEATURES, accuracy_threshold, "regression", verbose=verbose
    )

    india_candidates = [
        feature
        for feature in INDIA_BINARY_FEATURES
        if not (drop_prev_pregnancy and feature in PREV_PREGNANCY_FEATURES)
    ]
    india_binary, removed_india_binary = select_features_with_threshold(
        india_candidates, accuracy_threshold, "binary", verbose=verbose
    )
    india_regression, removed_india_regression = select_features_with_threshold(
        INDIA_REGRESSION_FEATURES, accuracy_threshold, "regression", verbose=verbose
    )

    return CrossDomainFeatures(
        msia_multiclass=msia_multiclass,
        msia_regression=msia_regression,
        india_binary=india_binary,
        india_regression=india_regression,
        removed_msia_multiclass=removed_msia_multiclass,
        removed_msia_regression=removed_msia_regression,
        removed_india_binary=removed_india_binary,
        removed_india_regression=removed_india_regression,
    )


# ── Per-fold data preparation ────────────────────────────────────────────────


@dataclass
class UnifiedFold:
    """Everything one fold of the unified pipeline needs to fit and score a model."""

    fold: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    raw_train_df: pd.DataFrame
    raw_test_df: pd.DataFrame
    features: list
    categorical_features: list
    continuous_features: list
    country_arr: np.ndarray
    cluster_ids: np.ndarray | None
    std_or_min: pd.Series
    mean_or_max: pd.Series
    validation_df: pd.DataFrame | None = None
    msia_train_df: pd.DataFrame | None = None
    india_train_df: pd.DataFrame | None = None
    label: str = LABEL

    @property
    def train_X(self):
        """Training design matrix in the canonical feature order."""
        return self.train_df[self.features]

    @property
    def train_Y(self):
        """Training labels."""
        return self.train_df[self.label]

    @property
    def test_X(self):
        """Test design matrix in the canonical feature order."""
        return self.test_df[self.features]

    @property
    def test_Y(self):
        """Test labels."""
        return self.test_df[self.label]


def _split_with_optional_validation(df, df_add_on, fold, id_exists, num_of_folds,
                                    with_validation):
    """Fold-wise split, optionally holding out the next fold for validation."""
    if not with_validation:
        train_df, test_df = process_raw_train_and_test_df(
            df, df_add_on, fold, id_exists=id_exists
        )
        return train_df, test_df, None

    validation_fold = (fold + 1) % num_of_folds
    train_df, test_df = process_raw_train_and_test_df(
        df[df["fold"] != validation_fold], df_add_on, fold, id_exists=id_exists
    )
    _, validation_df = process_raw_train_and_test_df(
        df, df_add_on, validation_fold, id_exists=id_exists
    )
    return train_df, test_df, validation_df


def _drop_features(frames, features):
    """Drop ``features`` in place from every frame that carries them."""
    for frame in frames:
        for feature in features:
            if feature in frame.columns:
                frame.drop(feature, axis=1, inplace=True)


def _descale_for_saving(df, categorical_features, continuous_features,
                        std_or_min, mean_or_max):
    """Undo standardisation so the persisted split CSVs are in clinical units."""
    out = df.copy()
    present = [c for c in continuous_features if c in out.columns]
    if present:
        out[present] = descale_feature(out[present], std_or_min, mean_or_max)
    for column in categorical_features:
        if column in out.columns:
            out[column] = out[column].astype(int)
    return out


def build_harmonized_folds(
    msia_ds,
    india_ds,
    num_of_folds=N_FOLDS_CV,
    label=LABEL,
    accuracy_threshold=0.8,
    drop_prev_pregnancy=False,
    train_source=None,
    train_scale_factor=1.0,
    scale_seed=None,
    smoting=True,
    undersampling=False,
    with_validation=False,
    verbose_selection=False,
    msia_ds_full=None,
    india_ds_full=None,
    external_test_fold=None,
    selected_features=None,
    feature_subset=None,
    fit_imputers_per_fold=True,
    imputer_seed=SEED,
):
    """Yield one fully prepared :class:`UnifiedFold` per cross-validation fold."""
    if smoting and undersampling:
        raise ValueError("SMOTE and undersampling cannot be used concurrently.")
    if external_test_fold is not None and (msia_ds_full is None or india_ds_full is None):
        raise ValueError("external_test_fold requires msia_ds_full and india_ds_full")

    msia_df, msia_df_add_on, *_ = separate_df_and_df_add_on(
        msia_ds, label, id_exists=True
    )
    india_df, india_df_add_on, *_ = separate_df_and_df_add_on(
        india_ds, label, id_exists=False
    )

    if external_test_fold is not None:
        msia_df_all, msia_add_all, *_ = separate_df_and_df_add_on(
            msia_ds_full, label, id_exists=True
        )
        india_df_all, india_add_all, *_ = separate_df_and_df_add_on(
            india_ds_full, label, id_exists=False
        )

    selection = select_cross_domain_features(
        accuracy_threshold,
        drop_prev_pregnancy,
        verbose=verbose_selection,
        selected_features=selected_features,
    )
    print(
        f"Harmonized feature space: Malaysia->India {selection.msia_impute}, "
        f"India->Malaysia {selection.india_impute}"
    )

    for fold in range(num_of_folds):
        print(f"\n{'=' * 60}\nFold {fold}/{num_of_folds - 1}\n{'=' * 60}")

        msia_train_df, msia_test_df, msia_validation_df = _split_with_optional_validation(
            msia_df, msia_df_add_on, fold, True, num_of_folds, with_validation
        )
        india_train_df, india_test_df, india_validation_df = (
            _split_with_optional_validation(
                india_df, india_df_add_on, fold, False, num_of_folds, with_validation
            )
        )

        if external_test_fold is not None:
            _, msia_test_df = process_raw_train_and_test_df(
                msia_df_all, msia_add_all, external_test_fold, id_exists=True
            )
            _, india_test_df = process_raw_train_and_test_df(
                india_df_all, india_add_all, external_test_fold, id_exists=False
            )

        # Training-set-size ablation:
        if train_scale_factor is not None and train_scale_factor < 1.0:
            msia_train_df = scale_training_partition(
                msia_train_df,
                train_scale_factor,
                label,
                random_state=(None if scale_seed is None else scale_seed + fold),
            )
            india_train_df = scale_training_partition(
                india_train_df,
                train_scale_factor,
                label,
                random_state=(None if scale_seed is None else scale_seed + 100 + fold),
            )

        msia_frames = [msia_train_df, msia_test_df]
        india_frames = [india_train_df, india_test_df]
        if with_validation:
            msia_frames.append(msia_validation_df)
            india_frames.append(india_validation_df)
        for frame in msia_frames + india_frames:
            cast_common_types(frame)

        # Cross-domain imputation, both directions.
        fold_imputers = {}
        if fit_imputers_per_fold and (selection.msia_impute or selection.india_impute):
            fold_imputers = fit_fold_imputers(
                {"malaysia": msia_train_df, "india": india_train_df},
                selection.imputation_targets,
                seed=imputer_seed,
                verbose=verbose_selection,
            )

        india_train_df = impute_df(
            india_train_df,
            selection.msia_impute,
            multiclass_features=selection.msia_multiclass,
            regression_features=selection.msia_regression,
            imputers=fold_imputers,
        )
        india_test_df = impute_df(
            india_test_df,
            selection.msia_impute,
            multiclass_features=selection.msia_multiclass,
            regression_features=selection.msia_regression,
            imputers=fold_imputers,
        )
        msia_train_df = impute_df(
            msia_train_df,
            selection.india_impute,
            binaryclass_features=selection.india_binary,
            regression_features=selection.india_regression,
            imputers=fold_imputers,
        )
        msia_test_df = impute_df(
            msia_test_df,
            selection.india_impute,
            binaryclass_features=selection.india_binary,
            regression_features=selection.india_regression,
            imputers=fold_imputers,
        )
        if with_validation:
            india_validation_df = impute_df(
                india_validation_df,
                selection.msia_impute,
                multiclass_features=selection.msia_multiclass,
                regression_features=selection.msia_regression,
                imputers=fold_imputers,
            )
            msia_validation_df = impute_df(
                msia_validation_df,
                selection.india_impute,
                binaryclass_features=selection.india_binary,
                regression_features=selection.india_regression,
                imputers=fold_imputers,
            )

        msia_frames = [msia_train_df, msia_test_df]
        india_frames = [india_train_df, india_test_df]
        if with_validation:
            msia_frames.append(msia_validation_df)
            india_frames.append(india_validation_df)

        _drop_features(msia_frames, selection.removed_from_malaysia)
        _drop_features(india_frames, selection.removed_from_india)
        if drop_prev_pregnancy:
            _drop_features(india_frames, PREV_PREGNANCY_FEATURES)

        for frame in msia_frames + india_frames:
            frame.reset_index(drop=True, inplace=True)

        if set(msia_train_df.columns) != set(india_train_df.columns):
            raise ValueError(
                "Malaysia and India training frames must share the same columns; "
                f"Malaysia-only: {set(msia_train_df.columns) - set(india_train_df.columns)}, "
                f"India-only: {set(india_train_df.columns) - set(msia_train_df.columns)}"
            )

        # Restrict the TRAINING rows to one country; the test set stays combined.
        if train_source == "malaysia":
            train_df = msia_train_df.copy()
        elif train_source == "india":
            train_df = india_train_df.copy()
        elif train_source is None:
            train_df = pd.concat([msia_train_df, india_train_df], axis=0)
        else:
            raise ValueError(
                f"train_source must be 'malaysia', 'india' or None, got {train_source!r}"
            )
        train_df = train_df.reset_index(drop=True)
        train_df[label] = train_df[label].astype(int)

        if with_validation:
            if train_source == "malaysia":
                validation_df = msia_validation_df.copy()
            elif train_source == "india":
                validation_df = india_validation_df.copy()
            else:
                validation_df = pd.concat(
                    [msia_validation_df, india_validation_df], axis=0
                )
            validation_df = validation_df.reset_index(drop=True)
            validation_df[label] = validation_df[label].astype(int)
        else:
            validation_df = None

        # Malaysia rows first, India second:
        msia_test_df[_COUNTRY_COLUMN] = MALAYSIA
        india_test_df[_COUNTRY_COLUMN] = INDIA
        # Pregnancy identifiers for the Malaysian test rows, taken from the frame
        # and fold this test partition was actually built from. Each Indian record
        # is a single pregnancy, so a synthetic per-row key is correct there.
        if external_test_fold is not None:
            id_source, id_fold = msia_df_all, external_test_fold
        else:
            id_source, id_fold = msia_df, fold
        msia_test_ids = id_source.loc[id_source["fold"] == id_fold, "id"].to_numpy()
        if len(msia_test_ids) == len(msia_test_df):
            msia_test_df[_CLUSTER_COLUMN] = [f"MY:{i}" for i in msia_test_ids]
        else:
            msia_test_df[_CLUSTER_COLUMN] = [
                f"MY:row{i}" for i in range(len(msia_test_df))
            ]
            print(
                f"  [warn] {len(msia_test_ids)} Malaysian ids for "
                f"{len(msia_test_df)} test rows; falling back to per-scan clusters."
            )
        india_test_df[_CLUSTER_COLUMN] = [f"IN:{i}" for i in range(len(india_test_df))]

        test_df = pd.concat([msia_test_df, india_test_df], axis=0).reset_index(drop=True)
        test_df[label] = test_df[label].astype(int)

        if test_df["ga"].max() > 300:
            raise ValueError("ga should not be larger than 300")

        raw_train_df = train_df.copy()
        raw_test_df = test_df.drop(columns=[_COUNTRY_COLUMN, _CLUSTER_COLUMN])

        # Resampling of the TRAINING rows only.
        X = train_df.drop(label, axis=1)
        y = train_df[label]
        # Index against X (label already dropped) so the SMOTENC categorical indices
        # stay valid regardless of where `label` sits in train_df.
        column_index = {col: idx for idx, col in enumerate(X.columns)}
        combined_categorical = (
            selection.india_binary + selection.msia_multiclass + ["gender"]
        )
        categorical_features = [
            col for col in combined_categorical if col in train_df.columns
        ]
        categorical_indices = [column_index[col] for col in categorical_features]
        continuous_features = [
            col for col in train_df.columns if col not in categorical_features
        ]
        if label in continuous_features:
            continuous_features.remove(label)

        smote = None
        if smoting:
            smote = SMOTENC(
                sampling_strategy="auto",
                categorical_features=categorical_indices,
                random_state=SEED,
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            train_df = pd.concat([X_resampled, y_resampled], axis=1)
        if undersampling:
            undersample = RandomUnderSampler(sampling_strategy="majority", random_state=SEED)
            X_resampled, y_resampled = undersample.fit_resample(X, y)
            train_df = pd.concat([X_resampled, y_resampled], axis=1)

        # The DNN pipeline keeps per-country resampled copies purely as a size guardrail
        # on the pooled training set.
        msia_only_train = india_only_train = None
        if with_validation and smoting:
            msia_X, msia_y = msia_train_df.drop(label, axis=1), msia_train_df[label]
            india_X, india_y = india_train_df.drop(label, axis=1), india_train_df[label]
            msia_resampled_X, msia_resampled_y = smote.fit_resample(msia_X, msia_y)
            msia_only_train = pd.concat([msia_resampled_X, msia_resampled_y], axis=1)
            india_resampled_X, india_resampled_y = smote.fit_resample(india_X, india_y)
            india_only_train = pd.concat([india_resampled_X, india_resampled_y], axis=1)

        for feature in categorical_features:
            train_df[feature] = train_df[feature].astype(int)
            test_df[feature] = test_df[feature].astype(int)

        # Guardrail: drop physiologically impossible rows before scaling.
        remove_illogical_values(train_df)
        remove_illogical_values(test_df)
        if with_validation:
            remove_illogical_values(validation_df)
            if msia_only_train is not None:
                remove_illogical_values(msia_only_train)
                remove_illogical_values(india_only_train)
                if train_source is None and (
                    len(msia_only_train) + len(india_only_train) != len(train_df)
                ):
                    raise ValueError(
                        "Malaysia + India training rows do not add up to the pooled "
                        "training set."
                    )

        country_arr = test_df.pop(_COUNTRY_COLUMN).to_numpy(dtype=int)
        cluster_ids = test_df.pop(_CLUSTER_COLUMN).to_numpy()

        train_df, test_df, validation_df, std_or_min, mean_or_max = (
            scale_sample_train_and_test_df(
                train_df,
                test_df,
                validation_df,
                categorical_features,
                continuous_features,
            )
        )

        # Scaling reorders the columns, so the feature order is recomputed here.
        features = [
            col for col in train_df.columns if col not in continuous_features
        ] + continuous_features
        features.remove(label)

        for feature in categorical_features:
            train_df[feature] = train_df[feature].astype(int)
            test_df[feature] = test_df[feature].astype(int)
            if validation_df is not None:
                validation_df[feature] = validation_df[feature].astype(int)

        train_df[label] = train_df[label].astype(int)
        test_df[label] = test_df[label].astype(int)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        if feature_subset is not None:
            missing = [c for c in feature_subset if c not in test_df.columns]
            if missing:
                raise ValueError(
                    "feature_subset asks for column(s) the prepared fold does not "
                    f"have: {missing}. Available: {sorted(test_df.columns)}"
                )
            features = list(feature_subset)
            categorical_features = [c for c in categorical_features if c in features]
            continuous_features = [c for c in continuous_features if c in features]

        yield UnifiedFold(
            fold=fold,
            train_df=train_df,
            test_df=test_df,
            raw_train_df=raw_train_df,
            raw_test_df=raw_test_df,
            features=features,
            categorical_features=categorical_features,
            continuous_features=continuous_features,
            country_arr=country_arr,
            cluster_ids=cluster_ids,
            std_or_min=std_or_min,
            mean_or_max=mean_or_max,
            validation_df=validation_df,
            msia_train_df=msia_only_train,
            india_train_df=india_only_train,
            label=label,
        )


# ── Shared result accumulation ───────────────────────────────────────────────


def _new_train_test_dict():
    """Empty accumulator matching the keys `save_train_test_set` understands."""
    return {
        "training_set": [],
        "testing_set": [],
        "validation_set": [],
        "generated_set": [],
        "raw_training_set": [],
        "raw_testing_set": [],
    }


def _save_scaling_values(scaling_numbers, download_path):
    """Persist the per-fold standardisation statistics for later inference."""
    scaling_save_path = f"{download_path}/scaling/value.csv"
    os.makedirs(os.path.dirname(scaling_save_path), exist_ok=True)
    pd.DataFrame(scaling_numbers).to_csv(scaling_save_path, index=False)


def _save_best_params(best_params, download_path, model_type, fold):
    """Write the winning GridSearchCV configuration for one fold."""
    path = f"{download_path}/model_parameters/{model_type}_best_params_{fold}.yaml"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        yaml.dump(best_params, handle)


def _print_fold_metrics(overall_cm, acc, roc_auc, f1, prec, rec):
    """Print the per-fold summary the original scripts logged."""
    ppv, npv, sensitivity, specificity = calculate_metrics(overall_cm)
    print(f"PPV: {ppv:.4f}")
    print(f"NPV: {npv:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"OOF ACC: {np.mean(acc)}")
    print(f"OOF ROC AUC Score: {np.mean(roc_auc)}")
    print(f"OOF F1 Score: {np.mean(f1):.4f}")
    print(f"OOF Prec Score: {np.mean(prec):.4f}")
    print(f"OOF Recall Score: {np.mean(rec):.4f}")


def _prediction_frame(test_X, test_Y, y_pred, features):
    """Rebuild the scaled test frame with the model's predictions attached."""
    return pd.DataFrame(
        data=np.concatenate(
            (np.asarray(test_X), test_Y.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1
        ),
        columns=features + ["Actual", "Prediction"],
    )


# ── CatBoost ─────────────────────────────────────────────────────────────────


def train_catboost_unified(
    msia_ds,
    india_ds,
    download_path,
    num_of_folds=N_FOLDS_CV,
    label=LABEL,
    accuracy_threshold=0.8,
    smoting=True,
    undersampling=False,
    drop_prev_pregnancy=False,
    train_source=None,
    train_scale_factor=1.0,
    scale_seed=None,
    selected_features=None,
):
    """Train the unified CatBoost classifier with per-fold GridSearchCV."""
    os.makedirs(download_path, exist_ok=True)

    models = []
    validation_balanced_accuracy, validation_oof_roc_auc = [], []
    validation_oof_f1, validation_oof_prec, validation_oof_rec = [], [], []
    overall_cm_list, roc_curve_result_list = [], []
    y_true_list, y_score_list = [], []
    train_test_dict = _new_train_test_dict()
    scaling_numbers = {"Fold": [], "std_or_min": [], "mean_or_max": []}
    country_results = []

    for prepared in build_harmonized_folds(
        msia_ds,
        india_ds,
        num_of_folds=num_of_folds,
        label=label,
        accuracy_threshold=accuracy_threshold,
        drop_prev_pregnancy=drop_prev_pregnancy,
        train_source=train_source,
        train_scale_factor=train_scale_factor,
        scale_seed=scale_seed,
        smoting=smoting,
        undersampling=undersampling,
        selected_features=selected_features,
    ):
        fold = prepared.fold
        scaling_numbers["Fold"].append(fold)
        scaling_numbers["std_or_min"].append(prepared.std_or_min)
        scaling_numbers["mean_or_max"].append(prepared.mean_or_max)
        train_test_dict["raw_training_set"].append(prepared.raw_train_df)
        train_test_dict["raw_testing_set"].append(prepared.raw_test_df)

        train_X, train_Y = prepared.train_X, prepared.train_Y
        test_X, test_Y = prepared.test_X, prepared.test_Y

        use_gpu = torch.cuda.is_available()
        cb = CatBoostClassifier(class_names=[0, 1], task_type="GPU" if use_gpu else "CPU")
        # n_jobs=1 on GPU (CatBoost parallelises internally); -1 on CPU.
        cb_grid = GridSearchCV(
            estimator=cb,
            param_grid=catboost_hyperparameters(),
            scoring=make_scorer(roc_auc_score),
            cv=5,
            n_jobs=1 if use_gpu else -1,
        )
        # No eval_set: CatBoost turns `use_best_model` on whenever one is supplied,
        # which would pick the stopping iteration on the very rows the fold is
        # scored against. The Methods require hyperparameters and stopping to be
        # chosen inside the training and validation folds only, so the inner
        # GridSearchCV cross-validation is the sole model-selection signal.
        cb_grid.fit(train_X, train_Y, verbose=100)

        train_pool = Pool(
            data=train_X, label=train_Y, cat_features=prepared.categorical_features
        )

        best_params = cb_grid.best_params_
        cb_best = CatBoostClassifier(
            **best_params, class_names=[0, 1], task_type="GPU" if use_gpu else "CPU"
        )
        cb_best.fit(train_pool, verbose=100)

        weights_dir = f"{download_path}/model_weights"
        os.makedirs(weights_dir, exist_ok=True)
        cb_best.save_model(f"{weights_dir}/model_{fold}")
        _save_best_params(best_params, download_path, "catboost", fold)

        y_pred = np.array(cb_best.predict(prepared.test_df[prepared.features]))
        y_pred_prob = np.array(
            [pred[1] for pred in cb_best.predict_proba(prepared.test_df[prepared.features])]
        )
        test_Y = np.array(test_Y)

        acc, roc_auc, f1, prec, rec, roc_curve_result = calc_metrics_with_ci(
            test_Y,
            y_pred,
            y_pred_prob,
            metrics=METRICS_LIST,
            download_path=download_path,
            fold=fold,
        )
        overall_cm = confusion_matrix(test_Y, y_pred)

        split_metrics = evaluate_splits(test_Y, y_pred, y_pred_prob, prepared.country_arr)
        append_fold_rows(country_results, split_metrics, "catboost", fold)

        compute_and_plot_permutation_importance(
            cb_best, test_X, test_Y, prepared.features, fold, download_path, "catboost"
        )
        _print_fold_metrics(overall_cm, acc, roc_auc, f1, prec, rec)

        prediction_df = _prediction_frame(test_X, test_Y, y_pred, prepared.features)
        train_test_dict["training_set"].append(
            _descale_for_saving(
                prepared.train_df,
                prepared.categorical_features,
                prepared.continuous_features,
                prepared.std_or_min,
                prepared.mean_or_max,
            )
        )
        train_test_dict["testing_set"].append(
            _descale_for_saving(
                prediction_df,
                prepared.categorical_features,
                prepared.continuous_features,
                prepared.std_or_min,
                prepared.mean_or_max,
            )
        )

        overall_cm_list.append(overall_cm)
        roc_curve_result_list.append(roc_curve_result)
        y_true_list.append(test_Y)
        y_score_list.append(y_pred_prob)
        validation_balanced_accuracy.append(acc)
        validation_oof_roc_auc.append(roc_auc)
        validation_oof_f1.append(f1)
        validation_oof_prec.append(prec)
        validation_oof_rec.append(rec)
        models.append(cb_best)

    _save_scaling_values(scaling_numbers, download_path)
    save_country_results(
        country_results, download_path, prefix="country", model_name="catboost"
    )
    display_ml_metrics(
        models,
        num_of_folds,
        download_path,
        validation_balanced_accuracy,
        validation_oof_roc_auc,
        validation_oof_f1,
        validation_oof_prec,
        validation_oof_rec,
        catboost=True,
    )
    display_roc_curve_binary(
        roc_curve_result_list, y_true_list, y_score_list, download_path
    )
    display_cm(overall_cm_list, download_path)
    save_train_test_set(train_test_dict, download_path)


# ── Classical scikit-learn models ────────────────────────────────────────────


def _build_sklearn_model(model_type):
    """Instantiate the estimator for ``model_type``."""
    if model_type == "stacking":
        return build_stacking_classifier(cv=5), False

    grids = {
        "rf": (RandomForestClassifier, rf_hyperparameters),
        "lr": (LogisticRegression, lr_hyperparameters),
        "svc": (SVC, svc_hyperparameters),
    }
    if model_type not in grids:
        raise ValueError(
            f"Invalid model type {model_type!r}; choose from {SKLEARN_MODEL_TYPES}"
        )
    model_class, param_func = grids[model_type]
    base = model_class(probability=True) if model_type == "svc" else model_class()
    return GridSearchCV(base, param_func(), cv=5, scoring="roc_auc", n_jobs=-1), True


def train_sklearn_unified(
    msia_ds,
    india_ds,
    download_path,
    model_type="rf",
    num_of_folds=N_FOLDS_CV,
    label=LABEL,
    accuracy_threshold=0.8,
    smoting=True,
    undersampling=False,
    drop_prev_pregnancy=False,
    train_source=None,
    train_scale_factor=1.0,
    scale_seed=None,
    selected_features=None,
):
    """Train one of the unified classical classifiers (LR / RF / SVC / Stacking)."""
    if model_type not in SKLEARN_MODEL_TYPES:
        raise ValueError(
            f"Invalid model type {model_type!r}; choose from {SKLEARN_MODEL_TYPES}"
        )
    os.makedirs(download_path, exist_ok=True)
    print(f"\nSELECTED MODEL TYPE: {model_type}")

    models = []
    validation_balanced_accuracy, validation_oof_roc_auc = [], []
    validation_oof_f1, validation_oof_prec, validation_oof_rec = [], [], []
    overall_cm_list, roc_curve_result_list = [], []
    y_true_list, y_score_list = [], []
    train_test_dict = _new_train_test_dict()
    scaling_numbers = {"Fold": [], "std_or_min": [], "mean_or_max": []}
    country_results = []

    for prepared in build_harmonized_folds(
        msia_ds,
        india_ds,
        num_of_folds=num_of_folds,
        label=label,
        accuracy_threshold=accuracy_threshold,
        drop_prev_pregnancy=drop_prev_pregnancy,
        train_source=train_source,
        train_scale_factor=train_scale_factor,
        scale_seed=scale_seed,
        smoting=smoting,
        undersampling=undersampling,
        selected_features=selected_features,
    ):
        fold = prepared.fold
        scaling_numbers["Fold"].append(fold)
        scaling_numbers["std_or_min"].append(prepared.std_or_min)
        scaling_numbers["mean_or_max"].append(prepared.mean_or_max)
        train_test_dict["raw_training_set"].append(prepared.raw_train_df)
        train_test_dict["raw_testing_set"].append(prepared.raw_test_df)

        train_X, train_Y = prepared.train_X, prepared.train_Y
        test_X, test_Y = prepared.test_X, prepared.test_Y

        net, is_grid_search = _build_sklearn_model(model_type)
        print("Fitting into GridSearchCV Net")
        net.fit(train_X, train_Y)
        print("Fitted Successfully into GridSearchCV Net")

        if is_grid_search:
            best_params = net.best_params_
            net = net.best_estimator_
        else:
            best_params = net.get_params()
        _save_best_params(best_params, download_path, model_type, fold)

        weights_dir = f"{download_path}/model_weights"
        os.makedirs(weights_dir, exist_ok=True)
        with open(f"{weights_dir}/model_{fold}.pkl", "wb") as handle:
            pickle.dump(net, handle)

        y_pred = np.array(net.predict(test_X))
        y_pred_prob = np.array([pred[1] for pred in net.predict_proba(test_X)])
        test_Y = np.array(test_Y)

        acc, roc_auc, f1, prec, rec, roc_curve_result = calc_metrics_with_ci(
            test_Y,
            y_pred,
            y_pred_prob,
            metrics=METRICS_LIST,
            download_path=download_path,
            fold=fold,
        )
        overall_cm = confusion_matrix(test_Y, y_pred)

        split_metrics = evaluate_splits(test_Y, y_pred, y_pred_prob, prepared.country_arr)
        append_fold_rows(country_results, split_metrics, model_type, fold)

        compute_and_plot_permutation_importance(
            net, test_X, test_Y, prepared.features, fold, download_path, model_type
        )
        _print_fold_metrics(overall_cm, acc, roc_auc, f1, prec, rec)

        prediction_df = _prediction_frame(test_X, test_Y, y_pred, prepared.features)
        train_test_dict["training_set"].append(
            _descale_for_saving(
                prepared.train_df,
                prepared.categorical_features,
                prepared.continuous_features,
                prepared.std_or_min,
                prepared.mean_or_max,
            )
        )
        train_test_dict["testing_set"].append(
            _descale_for_saving(
                prediction_df,
                prepared.categorical_features,
                prepared.continuous_features,
                prepared.std_or_min,
                prepared.mean_or_max,
            )
        )

        overall_cm_list.append(overall_cm)
        roc_curve_result_list.append(roc_curve_result)
        y_true_list.append(test_Y)
        y_score_list.append(y_pred_prob)
        print(f"Balanced Accuracy: {acc}")
        validation_balanced_accuracy.append(acc)
        validation_oof_roc_auc.append(roc_auc)
        validation_oof_f1.append(f1)
        validation_oof_prec.append(prec)
        validation_oof_rec.append(rec)
        models.append(net)

    _save_scaling_values(scaling_numbers, download_path)
    save_country_results(
        country_results, download_path, prefix="country", model_name=model_type
    )
    display_ml_metrics(
        models,
        num_of_folds,
        download_path,
        validation_balanced_accuracy,
        validation_oof_roc_auc,
        validation_oof_f1,
        validation_oof_prec,
        validation_oof_rec,
    )
    display_roc_curve_binary(
        roc_curve_result_list, y_true_list, y_score_list, download_path
    )
    display_cm(overall_cm_list, download_path)
    save_train_test_set(train_test_dict, download_path)


# ── Neural network ───────────────────────────────────────────────────────────


def _country_probabilities(net, test_X):
    """Score the Malaysia-first test matrix in one forward pass."""
    net.eval()
    with torch.no_grad():
        logits = (
            net(torch.as_tensor(test_X, dtype=torch.float32, device=DEVICE))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    return 1.0 / (1.0 + np.exp(-logits))


def train_dnn_unified(
    msia_ds,
    india_ds,
    download_path,
    hyperparameters,
    num_of_folds=N_FOLDS_CV,
    label=LABEL,
    model_size="large",
    accuracy_threshold=0.8,
    smoting=True,
    undersampling=False,
    pytorch_balanced_sampling=False,
    skip_shap=False,
    drop_prev_pregnancy=False,
    train_source=None,
    train_scale_factor=1.0,
    scale_seed=None,
    selected_features=None,
):
    """Train the unified feed-forward neural network."""
    if model_size not in MODEL_SIZES:
        raise ValueError(
            f"Invalid model size {model_size!r}; choose from {sorted(MODEL_SIZES)}"
        )
    os.makedirs(download_path, exist_ok=True)

    (
        dropout_rate,
        layer_output_size,
        epochs,
        batch_size,
        learning_rate,
        weight_decay,
        l1_lambda,
    ) = hyperparameters
    min_learning_rate = min(1e-4, learning_rate / 10)
    loss_criteria = nn.BCEWithLogitsLoss()
    model_class = MODEL_SIZES[model_size]

    epoch_nums, training_loss, validation_loss = [], [], []
    training_balanced_accuracy, training_oof_acc = [], []
    training_oof_roc_auc, training_oof_f1 = [], []
    training_oof_prec, training_oof_rec = [], []
    training_overall_cm_list, training_roc_curve_result_list = [], []
    training_y_true_list, training_y_score_list = [], []

    validation_balanced_accuracy, validation_oof_acc = [], []
    validation_oof_roc_auc, validation_oof_f1 = [], []
    validation_oof_prec, validation_oof_rec = [], []
    models, overall_cm_list, roc_curve_result_list = [], [], []
    y_true_list, y_score_list = [], []

    shap_dict = new_shap_dict()
    train_test_dict = _new_train_test_dict()
    output_layer_dict_list, permutation_importance_list = [], []
    scaling_numbers = {"Fold": [], "std_or_min": [], "mean_or_max": []}
    country_results = []

    for prepared in build_harmonized_folds(
        msia_ds,
        india_ds,
        num_of_folds=num_of_folds,
        label=label,
        accuracy_threshold=accuracy_threshold,
        drop_prev_pregnancy=drop_prev_pregnancy,
        train_source=train_source,
        train_scale_factor=train_scale_factor,
        scale_seed=scale_seed,
        smoting=smoting,
        undersampling=undersampling,
        selected_features=selected_features,
        with_validation=True,
    ):
        fold = prepared.fold
        scaling_numbers["Fold"].append(fold)
        scaling_numbers["std_or_min"].append(prepared.std_or_min)
        scaling_numbers["mean_or_max"].append(prepared.mean_or_max)
        train_test_dict["raw_training_set"].append(prepared.raw_train_df)
        train_test_dict["raw_testing_set"].append(prepared.raw_test_df)

        features = prepared.features
        net = model_class(len(features), dropout_rate, layer_output_size).to(DEVICE)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=int(epochs / 4), eta_min=min_learning_rate
        )
        optimizer.zero_grad()

        train_X = prepared.train_df[features].values
        train_Y = prepared.train_df[label].values
        validation_X = prepared.validation_df[features].values
        validation_Y = prepared.validation_df[label].values
        test_X = prepared.test_df[features].values
        test_Y = prepared.test_df[label].values
        # Malaysia-first copy kept for the per-country forward pass.
        country_test_X, country_test_Y = test_X.copy(), test_Y.copy()

        train_loader, test_loader, validation_loader = convert_to_tensor_dataloader(
            train_X,
            train_Y,
            test_X,
            test_Y,
            batch_size,
            validation_X,
            validation_Y,
            pytorch_balance_sampling=pytorch_balanced_sampling,
        )

        fold_epochs, fold_train_loss, fold_val_loss = [], [], []
        fold_train_bacc, fold_train_acc, fold_train_roc_auc = [], [], []
        fold_train_f1, fold_train_prec, fold_train_rec = [], [], []
        fold_train_roc_curve, fold_train_cm = [], []
        fold_train_true, fold_train_prob = [], []
        train_df, validation_df = prepared.train_df, prepared.validation_df

        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}")
            (
                train_loss,
                train_balanced_accuracy,
                val_loss,
                _val_steps,
                train_oof_acc,
                train_oof_roc_auc,
                train_oof_f1,
                train_oof_prec,
                train_oof_rec,
                train_roc_curve_result,
                train_overall_cm,
                train_true,
                train_prob,
                train_df,
                validation_df,
            ) = train_dnn(
                net,
                train_loader,
                optimizer,
                loss_criteria,
                features,
                l1_lambda=l1_lambda,
                get_training_details=True,
                download_path=download_path,
                fold=fold,
                final_epoch=(epoch == epochs),
                validation_loader=validation_loader,
            )
            fold_epochs.append(epoch)
            fold_train_loss.append(train_loss)
            fold_val_loss.append(val_loss)
            fold_train_bacc.append(train_balanced_accuracy)
            fold_train_acc.append(train_oof_acc)
            fold_train_roc_auc.append(train_oof_roc_auc)
            fold_train_f1.append(train_oof_f1)
            fold_train_prec.append(train_oof_prec)
            fold_train_rec.append(train_oof_rec)
            fold_train_roc_curve.append(train_roc_curve_result)
            fold_train_cm.append(train_overall_cm)
            fold_train_true.append(train_true)
            fold_train_prob.append(train_prob)
            lr_scheduler.step()

        epoch_nums.append(fold_epochs)
        training_loss.append(fold_train_loss)
        validation_loss.append(fold_val_loss)
        training_balanced_accuracy.append(fold_train_bacc[-1])
        training_oof_acc.append(fold_train_acc[-1])
        training_oof_roc_auc.append(fold_train_roc_auc[-1])
        training_oof_f1.append(fold_train_f1[-1])
        training_oof_prec.append(fold_train_prec[-1])
        training_oof_rec.append(fold_train_rec[-1])
        training_roc_curve_result_list.append(fold_train_roc_curve[-1])
        training_overall_cm_list.append(fold_train_cm[-1])
        training_y_true_list.append(fold_train_true[-1])
        training_y_score_list.append(fold_train_prob[-1])

        (
            _loss,
            test_balanced_accuracy,
            test_oof_acc,
            test_oof_roc_auc,
            test_oof_f1,
            test_oof_prec,
            test_oof_rec,
            roc_curve_result,
            overall_cm,
            test_df,
            output_layer_dict,
            fold_test_Y,
            y_pred_prob,
        ) = test_dnn(
            net,
            test_loader,
            loss_criteria,
            features,
            final=True,
            model_size=model_size,
            get_all_layers_output=False,
            download_path=download_path,
            fold=fold,
        )

        train_test_dict["training_set"].append(
            _descale_for_saving(
                train_df,
                prepared.categorical_features,
                prepared.continuous_features,
                prepared.std_or_min,
                prepared.mean_or_max,
            )
        )
        train_test_dict["validation_set"].append(
            _descale_for_saving(
                validation_df,
                prepared.categorical_features,
                prepared.continuous_features,
                prepared.std_or_min,
                prepared.mean_or_max,
            )
        )
        train_test_dict["testing_set"].append(
            _descale_for_saving(
                test_df,
                prepared.categorical_features,
                prepared.continuous_features,
                prepared.std_or_min,
                prepared.mean_or_max,
            )
        )
        output_layer_dict_list.append(output_layer_dict)

        data, target = next(iter(test_loader))
        permutation_importance_df = compute_permutation_importance(
            net, data.to(DEVICE), target.to(DEVICE), roc_auc_score, features
        )
        print("Permutation Importance Dataframe: ", permutation_importance_df)
        permutation_importance_list.append(permutation_importance_df)

        if not skip_shap:
            perform_SHAP(test_loader, net, features, shap_dict)

        overall_cm_list.append(overall_cm)
        roc_curve_result_list.append(roc_curve_result)
        y_true_list.append(fold_test_Y)
        y_score_list.append(y_pred_prob)

        # Per-country evaluation needs a forward pass in Malaysia-first order, because
        # the DataLoader-driven pass above may have reordered rows.
        country_prob = _country_probabilities(net, country_test_X)
        country_pred = (country_prob >= 0.5).astype(int)
        split_metrics = evaluate_splits(
            np.asarray(country_test_Y).astype(int),
            country_pred,
            country_prob,
            prepared.country_arr,
        )
        append_fold_rows(country_results, split_metrics, "dnn", fold)

        validation_balanced_accuracy.append(test_balanced_accuracy)
        validation_oof_acc.append(test_oof_acc)
        validation_oof_roc_auc.append(test_oof_roc_auc)
        validation_oof_f1.append(test_oof_f1)
        validation_oof_prec.append(test_oof_prec)
        validation_oof_rec.append(test_oof_rec)

        weights_dir = f"{download_path}/model_weights"
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(net.state_dict(), f"{weights_dir}/model_{fold}.pth")
        models.append(net)

    _save_scaling_values(scaling_numbers, download_path)
    save_country_results(
        country_results, download_path, prefix="country", model_name="dnn"
    )

    display_metrics_updated(
        models,
        num_of_folds,
        download_path,
        training_balanced_accuracy,
        training_oof_acc,
        training_oof_roc_auc,
        training_oof_f1,
        training_oof_prec,
        training_oof_rec,
        validation_balanced_accuracy,
        validation_oof_acc,
        validation_oof_roc_auc,
        validation_oof_f1,
        validation_oof_prec,
        validation_oof_rec,
    )
    display_training_loss(epoch_nums, training_loss, download_path)
    display_validation_loss(epoch_nums, validation_loss, download_path)
    if shap_dict["shap_df"]:
        display_shap(shap_dict, download_path)
    display_cm(training_overall_cm_list, download_path, training=True)
    display_roc_curve_binary(
        training_roc_curve_result_list,
        training_y_true_list,
        training_y_score_list,
        download_path,
        training=True,
    )
    display_roc_curve_binary(
        roc_curve_result_list, y_true_list, y_score_list, download_path
    )
    display_cm(overall_cm_list, download_path)
    save_train_test_set(train_test_dict, download_path)
    save_output_layer_dict(output_layer_dict_list, download_path)
    save_permutation_importance_list(permutation_importance_list, download_path)
