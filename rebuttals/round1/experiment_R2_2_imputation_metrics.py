"""Per-feature imputation quality (appendix Table S2).

Run:
    python -m rebuttals.round1.experiment_R2_2_imputation_metrics
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV

from sga.config import (
    CHART,
    INDIA_BINARY_FEATURES,
    INDIA_REGRESSION_FEATURES,
    INDIA_SUBDIR,
    MALAYSIA_MULTICLASS_FEATURES,
    MALAYSIA_REGRESSION_FEATURES,
    MALAYSIA_SUBDIR,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.data.cleaning import remove_illogical_values
from sga.data.scaling import scale_feature_test, scale_feature_train
from sga.imputation.apply import IMPUTER_INPUT_FEATURES
from sga.imputation.registry import model_dir
from sga.models.hyperparameters import (
    catboost_hyperparameters,
    regression_catboost_hyperparameters,
)
from sga.pipeline.dataset import load_cohort

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND1_DIR / "R2_2_imputation_metrics"

# Imputer training uses a FIXED split:
VALIDATION_FOLD = 3
TESTING_FOLD = 4

# The only categorical column among the ten imputer inputs.
IMPUTER_CATEGORICAL = ["gender"]
IMPUTER_CONTINUOUS = [f for f in IMPUTER_INPUT_FEATURES if f not in IMPUTER_CATEGORICAL]

REGRESSION_TARGETS = set(MALAYSIA_REGRESSION_FEATURES) | set(INDIA_REGRESSION_FEATURES)
BINARY_TARGETS = set(INDIA_BINARY_FEATURES)
MULTICLASS_TARGETS = set(MALAYSIA_MULTICLASS_FEATURES) | {"m_ethnics"}

# Reported in the order the appendix table lists them.
MSIA_REGRESSION = ["afi", "psv", "ute_ari"]
MSIA_MULTICLASS = ["af", "placenta_site"]
INDIA_REGRESSION = ["umb_api", "m_height", "m_weight"]
INDIA_BINARY = [
    "last_preg_sga",
    "last_preg_fgr",
    "last_preg_normal",
    "prev_failed_preg",
    "high_risk_pe",
    "hypertension_0",
    "hypertension_1",
    "diabetes_0",
]

# SMOGN is applied to a continuous target only when it is imbalanced enough.
MALAYSIA_SMOGN_THRESHOLD = 0.1
INDIA_SMOGN_THRESHOLD = 0.3
MIN_TRAINING_ROWS = 5
MIN_SAMPLES_PER_CLASS = 10


def _task_type(label):
    """Classify an imputation target as regression, binary or multiclass."""
    if label in REGRESSION_TARGETS:
        return "regression"
    if label in BINARY_TARGETS:
        return "binary"
    if label in MULTICLASS_TARGETS:
        return "multiclass"
    raise ValueError(f"Unknown task type for label: {label}")


def drop_samples(df, label, number_of_samples=MIN_SAMPLES_PER_CLASS):
    """Drop target classes with fewer than ``number_of_samples`` rows."""
    counts = df[label].value_counts()
    classes_to_drop = counts[counts < number_of_samples].index.tolist()
    if classes_to_drop:
        df = df[~df[label].isin(classes_to_drop)]
    return df, classes_to_drop


def _prepare_frames(full_df, add_on, label, id_exists):
    """Build the fixed train / validation / test partitions for one imputer."""
    keep = IMPUTER_INPUT_FEATURES + [label]
    meta = ["fold"] + (["id"] if id_exists else [])
    columns = [c for c in keep + meta if c in full_df.columns]

    frame = full_df[columns].copy()
    add_on_frame = add_on[[c for c in columns if c in add_on.columns]].copy()

    train_df = frame[(frame["fold"] != TESTING_FOLD) & (frame["fold"] != VALIDATION_FOLD)]
    validation_df = frame[frame["fold"] == VALIDATION_FOLD]
    test_df = frame[frame["fold"] == TESTING_FOLD]

    if id_exists:
        merged = pd.merge(add_on_frame, test_df["id"], on="id", how="left", indicator=True)
        add_on_train = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
        train_df = pd.concat([add_on_train, train_df], ignore_index=True)
    else:
        train_df = pd.concat([train_df, add_on_frame], axis=0, ignore_index=True)

    partitions = []
    for part in (train_df, validation_df, test_df):
        part = part.drop(columns=[c for c in meta if c in part.columns])
        part = part.dropna(subset=[label]).reset_index(drop=True)
        part["gender"] = part["gender"].astype(int)
        if label in REGRESSION_TARGETS:
            part[label] = part[label].round(2)
        else:
            part[label] = part[label].astype(int)
        partitions.append(part)
    return partitions


def _regression_metrics(prefix, y_true, y_pred, result):
    """MAE / RMSE / R^2 for one partition."""
    result[f"{prefix}_MAE"] = mean_absolute_error(y_true, y_pred)
    result[f"{prefix}_RMSE"] = root_mean_squared_error(y_true, y_pred)
    result[f"{prefix}_R2"] = r2_score(y_true, y_pred)
    return (f"MAE={result[f'{prefix}_MAE']:.4f}  RMSE={result[f'{prefix}_RMSE']:.4f}  "
            f"R2={result[f'{prefix}_R2']:.4f}")


def _binary_metrics(prefix, y_true, y_pred, model, X, result):
    """Accuracy / balanced accuracy / F1 / precision / recall / AUROC."""
    result[f"{prefix}_Accuracy"] = accuracy_score(y_true, y_pred)
    result[f"{prefix}_Balanced_Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    result[f"{prefix}_F1"] = f1_score(y_true, y_pred, zero_division=0)
    result[f"{prefix}_Precision"] = precision_score(y_true, y_pred, zero_division=0)
    result[f"{prefix}_Recall"] = recall_score(y_true, y_pred, zero_division=0)
    result[f"{prefix}_ROC_AUC"] = np.nan
    if len(np.unique(y_true)) > 1:
        try:
            result[f"{prefix}_ROC_AUC"] = roc_auc_score(
                y_true, np.asarray(model.predict_proba(X))[:, 1])
        except Exception:  # noqa: BLE001 - a degenerate target must not stop the sweep
            result[f"{prefix}_ROC_AUC"] = np.nan
    return (f"Acc={result[f'{prefix}_Accuracy']:.4f}  "
            f"BalAcc={result[f'{prefix}_Balanced_Accuracy']:.4f}  "
            f"F1={result[f'{prefix}_F1']:.4f}  AUC={result[f'{prefix}_ROC_AUC']}")


def _multiclass_metrics(prefix, y_true, y_pred, model, X, result):
    """Accuracy, balanced accuracy and weighted F1 / precision / recall."""
    result[f"{prefix}_Accuracy"] = accuracy_score(y_true, y_pred)
    result[f"{prefix}_Balanced_Accuracy"] = balanced_accuracy_score(y_true, y_pred)
    result[f"{prefix}_F1_weighted"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0)
    result[f"{prefix}_Precision_weighted"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0)
    result[f"{prefix}_Recall_weighted"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0)
    try:
        result[f"{prefix}_ROC_AUC_ovr"] = roc_auc_score(
            y_true, model.predict_proba(X), multi_class="ovr")
    except Exception:  # noqa: BLE001
        result[f"{prefix}_ROC_AUC_ovr"] = np.nan
    return (f"Acc={result[f'{prefix}_Accuracy']:.4f}  "
            f"BalAcc={result[f'{prefix}_Balanced_Accuracy']:.4f}  "
            f"F1w={result[f'{prefix}_F1_weighted']:.4f}")


def train_and_evaluate_feature(label, country, chart=CHART, base_dir=None):
    """Refit one CatBoost imputer and score it on validation and test."""
    result = {"feature": label, "country": country}
    task = _task_type(label)
    result["task"] = task

    id_exists = country == "malaysia"
    subdir = MALAYSIA_SUBDIR if id_exists else INDIA_SUBDIR
    full_df, add_on = load_cohort(
        subdir, chart=chart, base_dir=base_dir, exclude_external_fold=False)

    if label not in full_df.columns:
        print(f"  [SKIP] {label} not in dataframe columns.")
        result["status"] = "skipped_not_in_columns"
        return result, None

    train_df, validation_df, test_df = _prepare_frames(full_df, add_on, label, id_exists)

    train_df, classes_to_drop = drop_samples(train_df, label, MIN_SAMPLES_PER_CLASS)
    train_df = train_df.reset_index(drop=True)
    if len(classes_to_drop) > 0:
        validation_df = validation_df[
            ~validation_df[label].isin(classes_to_drop)].reset_index(drop=True)
        test_df = test_df[~test_df[label].isin(classes_to_drop)].reset_index(drop=True)

    if task == "binary":
        if train_df[label].nunique() <= 1:
            print(f"  [SKIP] Only one class for {label}; ending imputation model.")
            result["status"] = "skipped_single_class"
            return result, None
        column_index = {col: idx for idx, col in enumerate(train_df.columns)}
        smote = SMOTENC(
            sampling_strategy="auto",
            categorical_features=[column_index[c] for c in IMPUTER_CATEGORICAL],
        )
        resampled_X, resampled_y = smote.fit_resample(
            train_df[IMPUTER_INPUT_FEATURES], train_df[label])
        train_df = pd.concat([resampled_X, resampled_y], axis=1)
    elif task == "regression":
        distribution = train_df[label].value_counts()
        imbalance_ratio = distribution.min() / distribution.max()
        threshold = (MALAYSIA_SMOGN_THRESHOLD if country == "malaysia"
                     else INDIA_SMOGN_THRESHOLD)
        if imbalance_ratio >= threshold:
            try:
                import smogn

                train_df = smogn.smoter(data=train_df, y=label)
            except Exception as error:  # noqa: BLE001
                print(f"  [WARN] SMOGN failed for {label}: {error}")

    remove_illogical_values(train_df)
    remove_illogical_values(validation_df)
    remove_illogical_values(test_df)

    train_df[IMPUTER_CONTINUOUS], std_or_min, mean_or_max = scale_feature_train(
        train_df[IMPUTER_CONTINUOUS], method="std")
    validation_df[IMPUTER_CONTINUOUS] = scale_feature_test(
        validation_df[IMPUTER_CONTINUOUS], std_or_min, mean_or_max, method="std")
    test_continuous = [c for c in IMPUTER_CONTINUOUS if c in test_df.columns]
    test_df[test_continuous] = scale_feature_test(
        test_df[test_continuous], std_or_min, mean_or_max, method="std")

    for frame in (train_df, validation_df, test_df):
        for column in IMPUTER_CATEGORICAL:
            if column in frame.columns:
                frame[column] = frame[column].astype(int)
        if task != "regression" and label in frame.columns:
            frame[label] = frame[label].astype(int)
        frame.reset_index(drop=True, inplace=True)

    train_X, train_Y = train_df[IMPUTER_INPUT_FEATURES], train_df[label]
    validation_X, validation_Y = validation_df[IMPUTER_INPUT_FEATURES], validation_df[label]
    test_X, test_Y = test_df[IMPUTER_INPUT_FEATURES], test_df[label]

    if len(train_df) < MIN_TRAINING_ROWS:
        print(f"  [SKIP] Not enough training data for {label}.")
        result["status"] = "skipped_insufficient_data"
        return result, None

    if task == "binary":
        search = GridSearchCV(
            estimator=CatBoostClassifier(), param_grid=catboost_hyperparameters(),
            scoring=make_scorer(roc_auc_score), cv=5,
        )
        search.fit(train_X, train_Y, verbose=100)
        best_params = search.best_params_
        model = CatBoostClassifier(**best_params)
    elif task == "multiclass":
        model = CatBoostClassifier(
            loss_function="MultiClass", eval_metric="TotalF1",
            auto_class_weights="Balanced",
        )
        best_params = {}
    else:
        search = GridSearchCV(
            estimator=CatBoostRegressor(),
            param_grid=regression_catboost_hyperparameters(),
            scoring=make_scorer(root_mean_squared_error, greater_is_better=False), cv=5,
        )
        search.fit(train_X, train_Y, verbose=100)
        best_params = search.best_params_
        model = CatBoostRegressor(**best_params)

    model.fit(
        Pool(data=train_X, label=train_Y, cat_features=IMPUTER_CATEGORICAL), verbose=100)

    model_save_dir = model_dir(label)
    if best_params:
        parameters_dir = os.path.join(model_save_dir, "model_parameters")
        os.makedirs(parameters_dir, exist_ok=True)
        with open(os.path.join(parameters_dir, "catboost_best_params.yaml"), "w") as handle:
            yaml.dump(best_params, handle)

    weights_dir = os.path.join(model_save_dir, "model_weights")
    os.makedirs(weights_dir, exist_ok=True)
    model.save_model(os.path.join(weights_dir, "model_0"))
    print(f"  Model saved to: {weights_dir}/model_0")

    validation_pred = np.array(model.predict(validation_X)).flatten()
    validation_true = np.array(validation_Y)
    if task == "regression":
        line = _regression_metrics("val", validation_true, validation_pred, result)
    elif task == "binary":
        line = _binary_metrics("val", validation_true, validation_pred, model,
                               validation_X, result)
    else:
        line = _multiclass_metrics("val", validation_true, validation_pred, model,
                                   validation_X, result)
    print(f"  VAL  {label:15s} | {line}")

    if len(test_df) == 0:
        print(f"  [WARN] No test data for {label} (fold {TESTING_FOLD} empty after filtering).")
        result["status"] = "ok_no_test_data"
        return result, model

    test_pred = np.array(model.predict(test_X)).flatten()
    test_true = np.array(test_Y)
    if task == "regression":
        line = _regression_metrics("test", test_true, test_pred, result)
    elif task == "binary":
        line = _binary_metrics("test", test_true, test_pred, model, test_X, result)
    else:
        line = _multiclass_metrics("test", test_true, test_pred, model, test_X, result)
    print(f"  TEST {label:15s} | {line}")

    result["status"] = "ok"
    return result, model


REGRESSION_COLUMNS = [
    "feature", "country", "task",
    "val_MAE", "val_RMSE", "val_R2",
    "test_MAE", "test_RMSE", "test_R2", "status",
]
CLASSIFICATION_COLUMNS = [
    "feature", "country", "task",
    "val_Accuracy", "val_Balanced_Accuracy", "val_F1", "val_F1_weighted",
    "val_Precision", "val_Precision_weighted", "val_Recall", "val_Recall_weighted",
    "val_ROC_AUC", "val_ROC_AUC_ovr",
    "test_Accuracy", "test_Balanced_Accuracy", "test_F1", "test_F1_weighted",
    "test_Precision", "test_Precision_weighted", "test_Recall", "test_Recall_weighted",
    "test_ROC_AUC", "test_ROC_AUC_ovr", "status",
]


def run_experiment():
    """Refit and score every imputer, writing the appendix Table S2 CSVs."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Imputation model performance metrics")
    print(f"  Seed: {SEED}")
    print(f"  Metrics output: {SAVE_DIR}")

    all_results = []
    for header, country, features in (
        ("Malaysia-trained REGRESSION features", "malaysia", MSIA_REGRESSION),
        ("Malaysia-trained MULTICLASS features", "malaysia", MSIA_MULTICLASS),
        ("India-trained REGRESSION features", "india", INDIA_REGRESSION),
        ("India-trained BINARY features", "india", INDIA_BINARY),
    ):
        print(f"\n--- {header} ---")
        for label in features:
            print(f"\nTraining imputation model for: {label}")
            result, _ = train_and_evaluate_feature(label, country)
            all_results.append(result)

    pd.DataFrame(all_results).to_csv(SAVE_DIR / "all_imputation_metrics.csv", index=False)

    regression_results = [r for r in all_results if r.get("task") == "regression"]
    classification_results = [
        r for r in all_results if r.get("task") in ("binary", "multiclass")]

    if regression_results:
        frame = pd.DataFrame(regression_results)
        frame = frame[[c for c in REGRESSION_COLUMNS if c in frame.columns]]
        frame.to_csv(SAVE_DIR / "regression_imputation_metrics.csv", index=False)
        print("\nREGRESSION IMPUTATION METRICS")
        print(frame.to_string(index=False))

    if classification_results:
        frame = pd.DataFrame(classification_results)
        frame = frame[[c for c in CLASSIFICATION_COLUMNS if c in frame.columns]]
        frame.to_csv(SAVE_DIR / "classification_imputation_metrics.csv", index=False)
        print("\nCLASSIFICATION IMPUTATION METRICS")
        print(frame.to_string(index=False))

    print(f"\nAll results saved to: {SAVE_DIR}/")
    print("Done.")
    return all_results


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
