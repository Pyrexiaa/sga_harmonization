"""Per-fold refitting of the cross-domain imputation models."""

from __future__ import annotations

import os

import yaml
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from sga.config import COMMON_FEATURES, SEED
from sga.imputation.registry import model_dir
from sga.imputation.train_imputers import TARGET_SOURCE

#: Only ``gender`` is categorical among the ten common input features.
CATEGORICAL_INPUTS = ["gender"]

#: Hyperparameters that must not be forwarded from the persisted tuning run, because
#: they are set explicitly here or would fight the class weighting.
_BLOCKED_PARAMS = {
    "cat_features",
    "loss_function",
    "eval_metric",
    "auto_class_weights",
    "random_seed",
    "verbose",
    "class_weights",
    "scale_pos_weight",
}


def _tuned_params(feature):
    """Hyperparameters chosen for ``feature`` by ``scripts/02_train_imputers.py``."""
    try:
        base = model_dir(feature)
    except KeyError:
        return {}

    path = os.path.join(base, "model_parameters", "catboost_best_params.yaml")
    if not os.path.exists(path):
        return {}

    with open(path) as handle:
        params = yaml.safe_load(handle) or {}
    if not isinstance(params, dict):
        return {}
    return {k: v for k, v in params.items() if k not in _BLOCKED_PARAMS}


def _build_model(kind, feature, seed):
    """Instantiate the CatBoost estimator used to impute ``feature``."""
    params = _tuned_params(feature)
    if kind == "regression":
        return CatBoostRegressor(
            cat_features=CATEGORICAL_INPUTS, random_seed=seed, verbose=0, **params
        )
    if kind == "multiclass":
        return CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            auto_class_weights="Balanced",
            cat_features=CATEGORICAL_INPUTS,
            random_seed=seed,
            verbose=0,
            **params,
        )
    if kind == "binary":
        return CatBoostClassifier(
            auto_class_weights="Balanced",
            cat_features=CATEGORICAL_INPUTS,
            random_seed=seed,
            verbose=0,
            **params,
        )
    raise ValueError(f"Unknown imputer kind {kind!r}")


def _fit_one(frame, feature, kind, seed, verbose):
    """Fit a single imputer for ``feature`` on ``frame``."""
    if feature not in frame.columns:
        if verbose:
            print(f"    [{feature}] absent from the source training frame; not refitted.")
        return None

    missing_inputs = [f for f in COMMON_FEATURES if f not in frame.columns]
    if missing_inputs:
        raise KeyError(
            f"Cannot refit the {feature!r} imputer: the source training frame is "
            f"missing common input feature(s) {missing_inputs}."
        )

    usable = frame[frame[feature].notna()]
    if len(usable) == 0:
        if verbose:
            print(f"    [{feature}] no observed values in this fold; not refitted.")
        return None

    train_X = usable[COMMON_FEATURES].copy()
    train_X[CATEGORICAL_INPUTS] = train_X[CATEGORICAL_INPUTS].astype(int)
    train_Y = usable[feature]

    if kind != "regression":
        train_Y = train_Y.astype(int)
        if train_Y.nunique() <= 1:
            if verbose:
                print(f"    [{feature}] only one class in this fold; not refitted.")
            return None

    model = _build_model(kind, feature, seed)
    model.fit(Pool(data=train_X, label=train_Y, cat_features=CATEGORICAL_INPUTS), verbose=False)

    if verbose:
        print(
            f"    [{feature}] {kind} imputer refitted on {len(usable)} "
            f"{TARGET_SOURCE.get(feature, '?')} training rows"
        )
    return model


def fit_fold_imputers(source_frames, targets, seed=SEED, verbose=False):
    """Refit the cross-domain imputers on one fold's training partitions."""
    fitted = {}
    for feature, kind in targets:
        source = TARGET_SOURCE.get(feature)
        if source is None:
            raise KeyError(
                f"No source cohort registered for imputation target {feature!r}; "
                f"known targets: {sorted(TARGET_SOURCE)}"
            )
        if source not in source_frames:
            raise KeyError(
                f"Imputation target {feature!r} is fitted on the {source!r} cohort, "
                f"which was not supplied (got {sorted(source_frames)})."
            )

        model = _fit_one(source_frames[source], feature, kind, seed, verbose)
        if model is not None:
            fitted[feature] = model

    return fitted
