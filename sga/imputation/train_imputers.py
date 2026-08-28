"""Training of the per-feature cross-domain CatBoost imputation models."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import shap
import yaml
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from imblearn.over_sampling import SMOTENC
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sga.config import (
    COMMON_FEATURES,
    CONSTANT_ZERO_FEATURES,
    EXTERNAL_TEST_FOLD,
    IMPUTER_DIR,
    INDIA_BINARY_FEATURES,
    INDIA_REGRESSION_FEATURES,
    MALAYSIA_MULTICLASS_FEATURES,
    MALAYSIA_REGRESSION_FEATURES,
    SEED,
)
from sga.data.cleaning import remove_illogical_values
from sga.data.scaling import descale_feature, scale_feature_test, scale_feature_train
from sga.models.hyperparameters import (
    catboost_hyperparameters,
    improved_regression_catboost_hyperparameters,
)
from sga.reporting.artifacts import save_train_test_set
from sga.reporting.importance import catboost_f_importances
from sga.reporting.metrics_tables import (
    calc_metrics_updated,
    calculate_metrics,
    display_ml_metrics,
    display_ml_regression_metrics,
)
from sga.reporting.plots import display_cm, display_roc_curve

#: Fold used to score the imputers; folds 0-2 train, fold 4 stays untouched.
VALIDATION_FOLD = 3

#: Only ``gender`` is categorical among the ten common input features.
CATEGORICAL_INPUTS = ["gender"]

#: Targets that are effectively constant in their source cohort and are filled with a
#: constant instead of being modelled.
CONSTANT_TARGETS = set(CONSTANT_ZERO_FEATURES) | {"hypertension_1"}

BINARY_TARGETS = [f for f in INDIA_BINARY_FEATURES if f not in CONSTANT_TARGETS]
MULTICLASS_TARGETS = list(MALAYSIA_MULTICLASS_FEATURES)
REGRESSION_TARGETS = list(MALAYSIA_REGRESSION_FEATURES) + list(INDIA_REGRESSION_FEATURES)

#: Target feature -> cohort the imputer is fitted on.
TARGET_SOURCE = {
    **{f: "malaysia" for f in MALAYSIA_MULTICLASS_FEATURES + MALAYSIA_REGRESSION_FEATURES},
    **{f: "india" for f in BINARY_TARGETS + INDIA_REGRESSION_FEATURES},
}

#: Training order used when no explicit target list is given.
DEFAULT_TARGETS = (
    list(MALAYSIA_REGRESSION_FEATURES)
    + MULTICLASS_TARGETS
    + list(INDIA_REGRESSION_FEATURES)
    + BINARY_TARGETS
)

#: Number of RandomizedSearchCV draws over the continuous hyperparameter grid.
REGRESSION_SEARCH_ITERATIONS = 60

#: Classes with fewer than this many training rows are dropped before SMOTE.
MIN_CLASS_SUPPORT = 5


def target_kind(target):
    """Classify an imputation target."""
    if target in BINARY_TARGETS:
        return "binary"
    if target in MULTICLASS_TARGETS:
        return "multiclass"
    if target in REGRESSION_TARGETS:
        return "regression"
    raise KeyError(
        f"{target!r} is not a cross-domain imputation target. "
        f"Known targets: {sorted(BINARY_TARGETS + MULTICLASS_TARGETS + REGRESSION_TARGETS)}"
    )


def imputer_dir(target, source=None, base_dir=None):
    """Directory an imputer for ``target`` is written to."""
    source = source or TARGET_SOURCE[target]
    base = base_dir or IMPUTER_DIR
    return os.path.join(str(base), f"train_{source}_predict_{target}")


def prepare_frames(dfs, target, id_exists):
    """Restrict a cohort pair to the common input features plus the target."""
    df, add_on = dfs[0], dfs[1][dfs[0].columns]
    if target not in df.columns:
        raise KeyError(f"Target {target!r} is not present in this cohort's table.")

    features = list(COMMON_FEATURES)
    categorical_features = [f for f in features if f in CATEGORICAL_INPUTS]
    continuous_features = [f for f in features if f not in CATEGORICAL_INPUTS]

    keep = categorical_features + continuous_features + [target, "fold"]
    if id_exists:
        keep.append("id")

    df, add_on = df[keep].copy(), add_on[keep].copy()
    for frame in (df, add_on):
        frame[categorical_features] = frame[categorical_features].astype(int)
        if target in REGRESSION_TARGETS:
            frame[target] = frame[target].round(2)

    return df, add_on, categorical_features, continuous_features, features


def split_train_validation_test(df, add_on, id_exists):
    """Partition a cohort into training, validation and external-test frames."""
    development = ~df["fold"].isin([VALIDATION_FOLD, EXTERNAL_TEST_FOLD])
    train_df = df[development].reset_index(drop=True)
    validation_df = df[df["fold"] == VALIDATION_FOLD].reset_index(drop=True)
    test_df = df[df["fold"] == EXTERNAL_TEST_FOLD].reset_index(drop=True)

    if id_exists:
        held_out_ids = set(validation_df["id"].unique()) | set(test_df["id"].unique())
        add_on_train = add_on[~add_on["id"].isin(held_out_ids)].reset_index(drop=True)
        dropped = len(add_on) - len(add_on_train)
        if dropped:
            print(
                f"  Held-out pregnancies: dropped {dropped} add-on scan(s) belonging "
                f"to validation/external-fold pregnancies before imputer training."
            )
        train_df = pd.concat([add_on_train, train_df], ignore_index=True)
        drop = ["fold", "id"]
    else:
        train_df = pd.concat([train_df, add_on], axis=0, ignore_index=True)
        drop = ["fold"]

    return tuple(
        frame.drop(columns=drop).reset_index(drop=True)
        for frame in (train_df, validation_df, test_df)
    )


def _build_imputer(method, seed):
    """Instantiate the within-feature imputer used to complete model inputs."""
    if method == "iterative":
        return IterativeImputer(max_iter=10, random_state=seed)
    if method in ("mean", "median"):
        return SimpleImputer(strategy=method)
    if method == "mode":
        return SimpleImputer(strategy="most_frequent")
    raise ValueError(
        f"Unknown imputation method {method!r}; use iterative, mean, median or mode."
    )


def handle_missing_values(
    frames, features, categorical_features, target, method="iterative", seed=SEED
):
    """Drop rows with a missing target and complete the remaining input gaps."""
    train_df, validation_df, test_df = (
        frame.dropna(subset=[target]).reset_index(drop=True) for frame in frames
    )
    frames = [train_df, validation_df, test_df]

    present = [f for f in features if f in train_df.columns]
    if sum(int(frame[present].isna().sum().sum()) for frame in frames) == 0:
        print("  No missing input values; skipping within-feature imputation.")
        return train_df, validation_df, test_df

    continuous = [f for f in present if f not in categorical_features]
    categorical = [f for f in present if f in categorical_features]

    if continuous and any(frame[continuous].isna().any().any() for frame in frames):
        imputer = _build_imputer(method, seed).fit(train_df[continuous])
        for frame in frames:
            frame[continuous] = imputer.transform(frame[continuous])

    if categorical and any(frame[categorical].isna().any().any() for frame in frames):
        imputer = SimpleImputer(strategy="most_frequent").fit(train_df[categorical])
        for frame in frames:
            frame[categorical] = imputer.transform(frame[categorical]).astype(int)

    print(f"  Remaining missing training inputs: {int(train_df[present].isna().sum().sum())}")
    return train_df, validation_df, test_df


def _drop_rare_classes(df, target, minimum=MIN_CLASS_SUPPORT):
    """Remove target classes with too few training rows to model or resample."""
    counts = df[target].value_counts()
    rare = counts[counts < minimum].index
    if len(rare):
        print(f"  Dropping target classes with < {minimum} training rows: {list(rare)}")
    return df[~df[target].isin(rare)].reset_index(drop=True), rare


def _fit_binary(train_X, train_Y, categorical_features, path):
    """Grid-search and fit the binary CatBoost imputer."""
    search = GridSearchCV(
        estimator=CatBoostClassifier(cat_features=categorical_features),
        param_grid=catboost_hyperparameters(),
        scoring=make_scorer(roc_auc_score),
        cv=5,
    )
    search.fit(train_X, train_Y, verbose=100)
    _dump_params(search.best_params_, path)
    return CatBoostClassifier(**search.best_params_)


def _fit_regression(train_X, train_Y, categorical_features, path, seed):
    """Randomised-search and fit the continuous CatBoost imputer."""
    grid = improved_regression_catboost_hyperparameters()
    n_candidates = int(np.prod([len(values) for values in grid.values()]))
    search = RandomizedSearchCV(
        estimator=CatBoostRegressor(
            cat_features=categorical_features, random_seed=seed, verbose=0
        ),
        param_distributions=grid,
        n_iter=min(REGRESSION_SEARCH_ITERATIONS, n_candidates),
        scoring=make_scorer(root_mean_squared_error, greater_is_better=False),
        cv=5,
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(train_X, train_Y)
    _dump_params(search.best_params_, path)
    return CatBoostRegressor(**search.best_params_)


def _dump_params(best_params, path):
    """Persist the selected hyperparameters next to the model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        yaml.dump(best_params, handle)


def _score_classifier(model, validation_df, features, target, kind):
    """Validation metrics for a classification imputer."""
    y_true = np.asarray(validation_df[target])
    y_pred = np.asarray(model.predict(validation_df[features]))
    y_prob = np.asarray(model.predict_proba(validation_df[features]))
    multiclass = kind == "multiclass"

    scorers = [balanced_accuracy_score, roc_auc_score, f1_score,
               precision_score, recall_score]
    computable = (
        len(np.unique(y_true)) == y_prob.shape[1]
        if multiclass
        else len(np.unique(y_true)) > 1
    )
    if not computable:
        accuracy, f1, precision, recall = calc_metrics_updated(
            y_true, y_pred, y_prob,
            metrics=[s for s in scorers if s is not roc_auc_score],
            multiclass=multiclass,
        )
        return (accuracy, 0, f1, precision, recall), confusion_matrix(y_true, y_pred), None

    if multiclass:
        accuracy, auroc, f1, precision, recall = calc_metrics_updated(
            y_true, y_pred, y_prob, metrics=scorers, multiclass=True
        )
        curve = None
    else:
        accuracy, auroc, f1, precision, recall, curve = calc_metrics_updated(
            y_true, y_pred, y_prob[:, 1], metrics=[*scorers, roc_curve], multiclass=False
        )
    return (accuracy, auroc, f1, precision, recall), confusion_matrix(y_true, y_pred), curve


def _score_regressor(model, validation_df, features, target, train_Y):
    """Validation MAE/RMSE/R2 for a continuous imputer, against a median baseline."""
    y_true = np.asarray(validation_df[target], dtype=float)
    y_pred = np.asarray(model.predict(validation_df[features]), dtype=float)
    mae, rmse, r2 = calc_metrics_updated(
        y_true, y_pred, metrics=[mean_absolute_error, root_mean_squared_error, r2_score]
    )

    median = float(np.median(np.asarray(train_Y, dtype=float)))
    baseline_rmse = root_mean_squared_error(y_true, np.full_like(y_true, median))
    print(f"  median baseline: value={median:.4f} RMSE={baseline_rmse:.4f} R2=0.0000")
    if float(np.mean(r2)) <= 0.0:
        print(
            f"  WARNING: validation R2 ({float(np.mean(r2)):.4f}) does not beat median "
            f"imputation for {target!r}; the median is the better imputer here."
        )
    return mae, rmse, r2, y_pred


def train_feature_imputer(
    dfs,
    target,
    download_path,
    id_exists=True,
    impute_method="iterative",
    seed=SEED,
    skip_if_trained=True,
):
    """Fit, score and persist one cross-domain imputer."""
    kind = target_kind(target)
    os.makedirs(download_path, exist_ok=True)
    weights_dir = os.path.join(download_path, "model_weights")
    if skip_if_trained and os.path.isdir(weights_dir) and os.listdir(weights_dir):
        print(f"[{target}] weights already exist in {weights_dir}; skipping.")
        return False

    df, add_on, categorical_features, continuous_features, features = prepare_frames(
        dfs, target, id_exists
    )
    print(f"\n[{target}] {kind} imputer on {len(features)} common features: {features}")

    train_df, validation_df, test_df = split_train_validation_test(df, add_on, id_exists)
    splits = {"raw_training_set": [train_df.copy()], "raw_testing_set": [test_df.copy()]}

    train_df, validation_df, test_df = handle_missing_values(
        (train_df, validation_df, test_df),
        features, categorical_features, target, method=impute_method, seed=seed,
    )

    # Rare-class removal is a CLASSIFICATION guardrail:
    if kind == "regression":
        rare = pd.Index([])
    else:
        train_df, rare = _drop_rare_classes(train_df, target)

    if len(rare):
        # Drop the same classes from the held-out frames.
        validation_df = validation_df[~validation_df[target].isin(rare)].reset_index(drop=True)
        test_df = test_df[~test_df[target].isin(rare)].reset_index(drop=True)

    if kind == "binary":
        if train_df[target].nunique() <= 1:
            print(f"[{target}] only one class present in training; not modelled.")
            return False
        indices = [list(train_df.columns).index(c) for c in categorical_features]
        resampled_X, resampled_Y = SMOTENC(
            sampling_strategy="auto", categorical_features=indices, random_state=seed
        ).fit_resample(train_df[features], train_df[target])
        train_df = pd.concat([resampled_X, resampled_Y], axis=1)

    for frame in (train_df, validation_df, test_df):
        remove_illogical_values(frame)

    train_df[continuous_features], std_or_min, mean_or_max = scale_feature_train(
        train_df[continuous_features], method="std"
    )
    for frame in (validation_df, test_df):
        frame[continuous_features] = scale_feature_test(
            frame[continuous_features], std_or_min, mean_or_max, method="std"
        )
    for frame in (train_df, validation_df, test_df):
        frame[categorical_features] = frame[categorical_features].astype(int)
        if kind != "regression":
            frame[target] = frame[target].astype(int)

    train_X, train_Y = train_df[features], train_df[target]
    train_pool = Pool(data=train_X, label=train_Y, cat_features=categorical_features)

    params_path = os.path.join(download_path, "model_parameters", "catboost_best_params.yaml")
    if kind == "binary":
        model = _fit_binary(train_X, train_Y, categorical_features, params_path)
    elif kind == "multiclass":
        # The multiclass targets are small and heavily imbalanced; balanced class
        # weights on the fixed configuration outperform a grid search here.
        model = CatBoostClassifier(
            loss_function="MultiClass", eval_metric="TotalF1", auto_class_weights="Balanced"
        )
    else:
        model = _fit_regression(train_X, train_Y, categorical_features, params_path, seed)
    model.fit(train_pool, verbose=100)

    explainer = shap.TreeExplainer(model)
    catboost_f_importances(
        explainer, explainer.shap_values(train_pool), train_X, features, "0-2",
        download_path, multiclass=(kind == "multiclass"),
    )

    if kind == "regression":
        mae, rmse, r2, y_pred = _score_regressor(
            model, validation_df, features, target, train_Y
        )
        print(f"  MAE {np.mean(mae):.4f} | RMSE {np.mean(rmse):.4f} | R2 {np.mean(r2):.4f}")
        display_ml_regression_metrics(
            [model], 1, download_path, [mae], [rmse], [r2], catboost=True
        )
    else:
        (accuracy, auroc, f1, precision, recall), matrix, curve = _score_classifier(
            model, validation_df, features, target, kind
        )
        ppv, npv, sensitivity, specificity = calculate_metrics(matrix)
        print(
            f"  balanced acc {np.mean(accuracy):.4f} | AUROC {np.mean(auroc):.4f} | "
            f"F1 {np.mean(f1):.4f} | PPV {ppv:.4f} | NPV {npv:.4f} | "
            f"sens {sensitivity:.4f} | spec {specificity:.4f}"
        )
        if curve is not None:
            display_roc_curve([curve], download_path)
        display_ml_metrics(
            [model], 1, download_path, [accuracy], [auroc], [f1], [precision], [recall],
            catboost=True,
        )
        display_cm([matrix], download_path)
        y_pred = np.asarray(model.predict(validation_df[features]))

    predictions = validation_df[features].copy()
    predictions["Actual"] = np.asarray(validation_df[target])
    predictions["Prediction"] = np.asarray(y_pred).reshape(-1)

    for name, frame in (
        ("training_set", train_df), ("testing_set", test_df),
        ("validation_set", predictions),
    ):
        saved = frame.copy()
        saved[continuous_features] = descale_feature(
            saved[continuous_features], std_or_min, mean_or_max
        )
        splits[name] = [saved]
    save_train_test_set(splits, download_path)
    return True


def train_all_imputers(
    cohorts, targets=None, base_dir=None, impute_method="iterative", seed=SEED,
    skip_if_trained=True,
):
    """Train every requested cross-domain imputer in turn."""
    trained = []
    for target in targets or DEFAULT_TARGETS:
        source = TARGET_SOURCE[target]
        if source not in cohorts:
            print(f"[{target}] source cohort {source!r} not loaded; skipping.")
            continue
        if train_feature_imputer(
            cohorts[source],
            target,
            imputer_dir(target, source, base_dir),
            id_exists=(source == "malaysia"),
            impute_method=impute_method,
            seed=seed,
            skip_if_trained=skip_if_trained,
        ):
            trained.append(target)
    return trained
