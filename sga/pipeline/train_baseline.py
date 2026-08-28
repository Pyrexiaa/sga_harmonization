"""Country-specific baseline training (strategy (a) of the manuscript)."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from torch.optim.lr_scheduler import CosineAnnealingLR

from sga.config import (
    COMMON_FEATURES,
    INDIA_SUBDIR,
    LABEL,
    MALAYSIA_SUBDIR,
    N_FOLDS_TOTAL,
    SEED,
)
from sga.models.architecture import MODEL_SIZES
from sga.models.hyperparameters import catboost_hyperparameters
from sga.models.loops import test_dnn, train_dnn
from sga.models.torch_utils import DEVICE, convert_to_tensor_dataloader
from sga.pipeline.dataset import (
    load_cohort,
    process_raw_train_and_test_df,
    scale_sample_train_and_test_df,
    separate_df_and_df_add_on,
)
from sga.pipeline.train_unified import (
    METRICS_LIST,
    SKLEARN_MODEL_TYPES,
    _build_sklearn_model,
    _descale_for_saving,
    _new_train_test_dict,
    _print_fold_metrics,
    _save_best_params,
)
from sga.reporting.artifacts import save_output_layer_dict, save_train_test_set
from sga.reporting.importance import (
    compute_and_plot_permutation_importance,
    lr_f_importances,
    rf_f_importances,
    svc_f_importances,
)
from sga.reporting.metrics_tables import (
    calc_metrics_with_ci,
    display_metrics,
    display_ml_metrics,
)
from sga.reporting.plots import (
    display_cm,
    display_roc_curve_binary,
    display_training_loss,
    display_validation_loss,
)
from sga.reporting.shap_analysis import display_shap, new_shap_dict, perform_SHAP

#: The ten measurements available in BOTH cohorts, as enumerated in the Methods
#: ("Baseline Model Training":
BASELINE_COMMON_FEATURES = list(COMMON_FEATURES)

#: The narrower seven-feature set the earlier runs actually used.
LEGACY_BASELINE_FEATURES = ["ga", "gender", "hc", "ac", "fl", "m_age", "efw"]

COUNTRY_SUBDIRS = {"malaysia": MALAYSIA_SUBDIR, "india": INDIA_SUBDIR}


def load_country_dataset(country, chart=None, base_dir=None, label=LABEL,
                         exclude_external_fold=True):
    """Load one cohort restricted to the shared baseline feature set."""
    if country not in COUNTRY_SUBDIRS:
        raise ValueError(f"country must be one of {sorted(COUNTRY_SUBDIRS)}, got {country!r}")

    id_exists = country == "malaysia"
    kwargs = {"chart": chart} if chart else {}
    # Fold 4 is excluded, exactly as for the unified models.
    dfs = load_cohort(
        COUNTRY_SUBDIRS[country],
        base_dir=base_dir,
        exclude_external_fold=exclude_external_fold,
        **kwargs,
    )
    columns = BASELINE_COMMON_FEATURES + [label] + (["id"] if id_exists else []) + ["fold"]
    return [frame[columns].copy() for frame in dfs], id_exists


@dataclass
class BaselineFold:
    """One prepared fold of the single-country baseline pipeline."""

    fold: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    validation_df: pd.DataFrame
    raw_train_df: pd.DataFrame
    features: list
    categorical_features: list
    continuous_features: list
    std_or_min: pd.Series
    mean_or_max: pd.Series
    label: str = LABEL


def build_baseline_folds(
    dfs,
    num_of_folds=N_FOLDS_TOTAL,
    label=LABEL,
    id_exists=True,
    add_noise_features=None,
    sample_size=None,
    smoting=True,
    undersampling=False,
):
    """Yield one prepared :class:`BaselineFold` per cross-validation fold."""
    if smoting and undersampling:
        raise ValueError("SMOTE and undersampling cannot be used concurrently.")

    df, df_add_on, categorical_features, continuous_features, features = (
        separate_df_and_df_add_on(dfs, label, id_exists)
    )

    for fold in range(num_of_folds):
        print(f"\n{'=' * 60}\nFold {fold}/{num_of_folds - 1}\n{'=' * 60}")

        validation_fold = (fold + 1) % num_of_folds
        train_df, test_df = process_raw_train_and_test_df(
            df[df["fold"] != validation_fold],
            df_add_on,
            fold,
            id_exists=id_exists,
            add_noise_features=add_noise_features,
        )
        _, validation_df = process_raw_train_and_test_df(
            df, df_add_on, validation_fold, id_exists=id_exists
        )
        raw_train_df = train_df.copy()

        # NOTE (pre-existing behaviour, deliberately preserved):
        if smoting:
            categorical_indices = [
                list(train_df.columns).index(column)
                for column in categorical_features
                if column in train_df.columns
            ]
            smote = SMOTENC(
                sampling_strategy="auto",
                categorical_features=categorical_indices,
                random_state=SEED,
            )
            resampled_X, resampled_y = smote.fit_resample(
                train_df[features], train_df[label]
            )
            train_df = pd.concat([resampled_X, resampled_y], axis=1)
        elif undersampling:
            undersample = RandomUnderSampler(sampling_strategy="majority", random_state=SEED)
            resampled_X, resampled_y = undersample.fit_resample(
                train_df[features], train_df[label]
            )
            train_df = pd.concat([resampled_X, resampled_y], axis=1)

        train_df, test_df, validation_df, std_or_min, mean_or_max = (
            scale_sample_train_and_test_df(
                train_df,
                test_df,
                validation_df,
                categorical_features,
                continuous_features,
                sample_size=sample_size,
            )
        )

        for column in categorical_features:
            train_df[column] = train_df[column].astype(int)
            test_df[column] = test_df[column].astype(int)
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        yield BaselineFold(
            fold=fold,
            train_df=train_df,
            test_df=test_df,
            validation_df=validation_df,
            raw_train_df=raw_train_df,
            features=features,
            categorical_features=categorical_features,
            continuous_features=continuous_features,
            std_or_min=std_or_min,
            mean_or_max=mean_or_max,
            label=label,
        )


def _collect_split_frames(prepared, train_test_dict, test_frame=None):
    """Descale and stash this fold's train/validation/test frames for saving."""
    args = (
        prepared.categorical_features,
        prepared.continuous_features,
        prepared.std_or_min,
        prepared.mean_or_max,
    )
    train_test_dict["raw_training_set"].append(prepared.raw_train_df)
    train_test_dict["training_set"].append(
        _descale_for_saving(prepared.train_df, *args)
    )
    train_test_dict["testing_set"].append(
        _descale_for_saving(
            prepared.test_df if test_frame is None else test_frame, *args
        )
    )
    train_test_dict["validation_set"].append(
        _descale_for_saving(prepared.validation_df, *args)
    )


def train_baseline_catboost(
    dfs,
    download_path,
    num_of_folds=N_FOLDS_TOTAL,
    label=LABEL,
    id_exists=True,
    smoting=True,
    undersampling=False,
    sample_size=None,
    add_noise_features=None,
    pretrained_model=False,
):
    """Train the country-specific CatBoost baseline."""
    os.makedirs(download_path, exist_ok=True)

    models, overall_cm_list, roc_curve_result_list = [], [], []
    metrics = {key: [] for key in ("bacc", "roc_auc", "f1", "prec", "rec")}
    train_test_dict = _new_train_test_dict()

    for prepared in build_baseline_folds(
        dfs,
        num_of_folds=num_of_folds,
        label=label,
        id_exists=id_exists,
        add_noise_features=add_noise_features,
        sample_size=sample_size,
        smoting=smoting,
        undersampling=undersampling,
    ):
        fold, features = prepared.fold, prepared.features
        train_X = prepared.train_df[features]
        train_Y = prepared.train_df[label].astype(int)
        test_X = prepared.test_df[features]
        test_Y = prepared.test_df[label].astype(int)

        if pretrained_model:
            cb_best = CatBoostClassifier()
            cb_best.load_model(f"{download_path}/model_weights/model_{fold}")
        else:
            cb_grid = GridSearchCV(
                estimator=CatBoostClassifier(class_names=[0, 1]),
                param_grid=catboost_hyperparameters(),
                scoring=make_scorer(roc_auc_score),
                cv=5,
            )
            cb_grid.fit(train_X, train_Y, eval_set=(test_X, test_Y), verbose=100)

            best_params = cb_grid.best_params_
            cb_best = CatBoostClassifier(**best_params, class_names=[0, 1])
            cb_best.fit(
                Pool(
                    data=train_X,
                    label=train_Y,
                    cat_features=prepared.categorical_features,
                ),
                eval_set=Pool(
                    data=test_X,
                    label=test_Y,
                    cat_features=prepared.categorical_features,
                ),
                verbose=100,
            )
            _save_best_params(best_params, download_path, "catboost", fold)

        compute_and_plot_permutation_importance(
            cb_best, test_X, test_Y, features, fold, download_path, "catboost"
        )

        y_pred = np.array(cb_best.predict(test_X))
        y_pred_prob = np.array([p[1] for p in cb_best.predict_proba(test_X)])
        test_Y = np.array(test_Y)

        acc, roc_auc, f1, prec, rec, roc_result = calc_metrics_with_ci(
            test_Y,
            y_pred,
            y_pred_prob,
            metrics=METRICS_LIST,
            download_path=download_path,
            fold=fold,
            save=True,
        )
        overall_cm = confusion_matrix(test_Y, y_pred)
        _print_fold_metrics(overall_cm, acc, roc_auc, f1, prec, rec)

        _collect_split_frames(prepared, train_test_dict)
        overall_cm_list.append(overall_cm)
        roc_curve_result_list.append(roc_result)
        for key, value in zip(metrics, (acc, roc_auc, f1, prec, rec)):
            metrics[key].append(value)
        models.append(cb_best)

    display_ml_metrics(
        models,
        num_of_folds,
        download_path,
        metrics["bacc"],
        metrics["roc_auc"],
        metrics["f1"],
        metrics["prec"],
        metrics["rec"],
        catboost=True,
    )
    display_cm(overall_cm_list, download_path)
    save_train_test_set(train_test_dict, download_path)


def _baseline_feature_importance(model, model_type, test_X, test_Y, features,
                                 fold, download_path):
    """Save permutation importance plus the model-specific importance measure."""
    compute_and_plot_permutation_importance(
        model, test_X, test_Y, features, fold, download_path, model_type
    )
    if model_type == "rf":
        rf_f_importances(model.feature_importances_, features, fold, download_path)
    elif model_type == "lr":
        lr_f_importances(model.coef_[0], features, fold, download_path)
    elif model_type == "svc":
        from sklearn.inspection import permutation_importance

        svc_f_importances(
            permutation_importance(model, test_X, test_Y), features, fold, download_path
        )


def train_baseline_sklearn(
    dfs,
    download_path,
    model_type="rf",
    num_of_folds=N_FOLDS_TOTAL,
    label=LABEL,
    id_exists=True,
    smoting=True,
    undersampling=False,
    sample_size=None,
    add_noise_features=None,
    pretrained_model=False,
):
    """Train a country-specific classical baseline (LR / RF / SVC / Stacking)."""
    if model_type not in SKLEARN_MODEL_TYPES:
        raise ValueError(
            f"Invalid model type {model_type!r}; choose from {SKLEARN_MODEL_TYPES}"
        )
    print(f"\nSELECTED MODEL TYPE: {model_type}")
    os.makedirs(download_path, exist_ok=True)

    models, overall_cm_list, roc_curve_result_list = [], [], []
    y_true_list, y_score_list = [], []
    metrics = {key: [] for key in ("bacc", "roc_auc", "f1", "prec", "rec")}
    train_test_dict = _new_train_test_dict()

    for prepared in build_baseline_folds(
        dfs,
        num_of_folds=num_of_folds,
        label=label,
        id_exists=id_exists,
        add_noise_features=add_noise_features,
        sample_size=sample_size,
        smoting=smoting,
        undersampling=undersampling,
    ):
        fold, features = prepared.fold, prepared.features
        train_X = prepared.train_df[features].values
        train_Y = prepared.train_df[label].values
        test_X = prepared.test_df[features].values
        test_Y = prepared.test_df[label].values

        if pretrained_model:
            with open(f"{download_path}/model_weights/model_{fold}.pkl", "rb") as handle:
                net = pickle.load(handle)
        else:
            net, is_grid_search = _build_sklearn_model(model_type)
            net.fit(train_X, train_Y)
            if is_grid_search:
                best_params = net.best_params_
                net = net.best_estimator_
            else:
                best_params = net.get_params()
            _save_best_params(best_params, download_path, model_type, fold)

        # Training-set metrics, written to the train confidence-interval table.
        train_pred = net.predict(train_X)
        train_pred_prob = np.array([p[1] for p in net.predict_proba(train_X)])
        calc_metrics_with_ci(
            np.array(train_Y),
            np.array(train_pred),
            train_pred_prob,
            metrics=METRICS_LIST,
            download_path=download_path,
            fold=fold,
            training=True,
        )

        y_pred = np.array(net.predict(test_X))
        y_pred_prob = np.array([p[1] for p in net.predict_proba(test_X)])
        test_Y = np.array(test_Y)

        acc, roc_auc, f1, prec, rec, roc_result = calc_metrics_with_ci(
            test_Y,
            y_pred,
            y_pred_prob,
            metrics=METRICS_LIST,
            download_path=download_path,
            fold=fold,
        )
        overall_cm = confusion_matrix(test_Y, y_pred)
        _print_fold_metrics(overall_cm, acc, roc_auc, f1, prec, rec)

        _baseline_feature_importance(
            net, model_type, test_X, test_Y, features, fold, download_path
        )

        _collect_split_frames(prepared, train_test_dict)
        overall_cm_list.append(overall_cm)
        roc_curve_result_list.append(roc_result)
        y_true_list.append(test_Y)
        y_score_list.append(y_pred_prob)
        for key, value in zip(metrics, (acc, roc_auc, f1, prec, rec)):
            metrics[key].append(value)
        models.append(net)

    display_ml_metrics(
        models,
        num_of_folds,
        download_path,
        metrics["bacc"],
        metrics["roc_auc"],
        metrics["f1"],
        metrics["prec"],
        metrics["rec"],
    )
    display_roc_curve_binary(
        roc_curve_result_list, y_true_list, y_score_list, download_path
    )
    display_cm(overall_cm_list, download_path)
    save_train_test_set(train_test_dict, download_path)


def train_baseline_dnn(
    dfs,
    download_path,
    hyperparameters,
    num_of_folds=N_FOLDS_TOTAL,
    label=LABEL,
    id_exists=True,
    model_size="large",
    smoting=True,
    undersampling=False,
    sample_size=None,
    add_noise_features=None,
    skip_shap=False,
):
    """Train the country-specific feed-forward neural-network baseline."""
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
    training_balanced_accuracy = []
    validation_balanced_accuracy, validation_oof_acc = [], []
    validation_oof_roc_auc, validation_oof_f1 = [], []
    validation_oof_prec, validation_oof_rec = [], []
    models, overall_cm_list, roc_curve_result_list = [], [], []
    y_true_list, y_score_list = [], []
    shap_dict = new_shap_dict()
    output_layer_dict_list = []
    train_test_dict = _new_train_test_dict()

    for prepared in build_baseline_folds(
        dfs,
        num_of_folds=num_of_folds,
        label=label,
        id_exists=id_exists,
        add_noise_features=add_noise_features,
        sample_size=sample_size,
        smoting=smoting,
        undersampling=undersampling,
    ):
        fold, features = prepared.fold, prepared.features
        net = model_class(len(features), dropout_rate, layer_output_size).to(DEVICE)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, epochs // 4), eta_min=min_learning_rate
        )

        train_loader, test_loader = convert_to_tensor_dataloader(
            prepared.train_df[features].values,
            prepared.train_df[label].values,
            prepared.test_df[features].values,
            prepared.test_df[label].values,
            batch_size,
        )

        fold_epochs, fold_train_loss, fold_val_loss, fold_train_bacc = [], [], [], []
        for epoch in range(1, epochs + 1):
            train_loss, train_bacc = train_dnn(
                net, train_loader, optimizer, loss_criteria, l1_lambda=l1_lambda
            )
            test_loss, *_ = test_dnn(
                net, test_loader, loss_criteria, features, model_size=model_size,
                download_path=download_path, fold=fold, save=False,
            )
            fold_epochs.append(epoch)
            fold_train_loss.append(train_loss)
            fold_train_bacc.append(train_bacc)
            fold_val_loss.append(test_loss)
            lr_scheduler.step()

        epoch_nums.append(fold_epochs)
        training_loss.append(fold_train_loss)
        validation_loss.append(fold_val_loss)
        training_balanced_accuracy.append(fold_train_bacc[-1])

        (
            _loss,
            test_bacc,
            test_acc,
            test_roc_auc,
            test_f1,
            test_prec,
            test_rec,
            roc_curve_result,
            overall_cm,
            test_result_df,
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
            download_path=download_path,
            fold=fold,
        )

        _collect_split_frames(prepared, train_test_dict, test_frame=test_result_df)
        output_layer_dict_list.append(output_layer_dict)
        if not skip_shap:
            perform_SHAP(test_loader, net, features, shap_dict)

        overall_cm_list.append(overall_cm)
        roc_curve_result_list.append(roc_curve_result)
        y_true_list.append(fold_test_Y)
        y_score_list.append(y_pred_prob)
        validation_balanced_accuracy.append(test_bacc)
        validation_oof_acc.append(test_acc)
        validation_oof_roc_auc.append(test_roc_auc)
        validation_oof_f1.append(test_f1)
        validation_oof_prec.append(test_prec)
        validation_oof_rec.append(test_rec)
        models.append(net)

    display_metrics(
        models,
        num_of_folds,
        download_path,
        training_balanced_accuracy,
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
    display_roc_curve_binary(
        roc_curve_result_list, y_true_list, y_score_list, download_path
    )
    display_cm(overall_cm_list, download_path)
    save_train_test_set(train_test_dict, download_path)
    save_output_layer_dict(output_layer_dict_list, download_path)
