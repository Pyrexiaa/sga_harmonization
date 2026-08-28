"""Feature-specific cross-domain imputation and imputation-quality gating."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from sga.config import (
    BINARY_AUROC_THRESHOLD,
    BINARY_F1_THRESHOLD,
    COMMON_FEATURES,
    CONSTANT_ZERO_FEATURES,
    MULTICLASS_AUROC_THRESHOLD,
    MULTICLASS_F1_THRESHOLD,
    PREV_PREGNANCY_FEATURES,
    REGRESSION_R2_THRESHOLD,
)
from sga.imputation.registry import model_dir

# Order expected by the pretrained imputers.
IMPUTER_INPUT_FEATURES = ["m_age", "hc", "ac", "fl", "efw", "ga", "gender", "bpd", "cpr", "ute_api"]
if set(IMPUTER_INPUT_FEATURES) != set(COMMON_FEATURES):
    # A plain assert would be skipped under `python -O`, silently letting the imputers
    # be fed a different feature space than they were trained on.
    raise ValueError(
        "IMPUTER_INPUT_FEATURES must be a permutation of config.COMMON_FEATURES; "
        f"only in IMPUTER_INPUT_FEATURES: {set(IMPUTER_INPUT_FEATURES) - set(COMMON_FEATURES)}, "
        f"only in COMMON_FEATURES: {set(COMMON_FEATURES) - set(IMPUTER_INPUT_FEATURES)}"
    )


def _load_imputer(feature, kind, fold=0):
    """Load the persisted development-block imputer for ``feature``."""
    path = f"{model_dir(feature)}/model_weights/model_{fold}"
    if kind == "binary":
        model = CatBoostClassifier()
    elif kind == "multiclass":
        model = CatBoostClassifier(
            loss_function="MultiClass", eval_metric="TotalF1", auto_class_weights="Balanced"
        )
    elif kind == "regression":
        model = CatBoostRegressor(verbose=0)
    else:
        raise ValueError(f"Unknown imputer kind {kind!r}")
    model.load_model(path)
    return model


def impute_df(
    df,
    impute_features,
    binaryclass_features=(),
    multiclass_features=(),
    regression_features=(),
    verbose=False,
    imputers=None,
):
    """Append cross-domain features to ``df`` using the cross-domain imputers."""
    binaryclass_features = set(binaryclass_features)
    multiclass_features = set(multiclass_features)
    regression_features = set(regression_features)
    imputers = imputers or {}

    out = df.copy()
    for feature in impute_features:
        if feature in df.columns:
            continue
        if feature in CONSTANT_ZERO_FEATURES:
            out[feature] = 0
            continue

        if feature in binaryclass_features:
            kind = "binary"
        elif feature in multiclass_features:
            kind = "multiclass"
        elif feature in regression_features:
            kind = "regression"
        else:
            raise ValueError(f"Feature {feature!r} has no imputation model type assigned")

        model = imputers.get(feature)
        if model is None:
            model = _load_imputer(feature, kind)

        inputs = df[IMPUTER_INPUT_FEATURES].copy()
        inputs["gender"] = inputs["gender"].astype(int)
        pool = Pool(data=inputs, cat_features=["gender"])
        values = np.asarray(model.predict(pool)).reshape(-1)
        if kind != "regression":
            values = values.astype(int)

        predictions = pd.DataFrame({feature: values}, index=df.index)
        out = pd.concat([out, predictions], axis=1)
        if verbose:
            source = "fold-refitted" if feature in imputers else "persisted"
            print(f"Imputed {feature} ({kind}, {source}):\n{predictions.value_counts()}")

    return out


def _mean_metric(metrics, column):
    """Mean of ``column`` across the recorded folds, or None when unavailable."""
    if column not in metrics.columns:
        return None
    values = pd.to_numeric(metrics[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _imputation_quality(feature, selection_type):
    """Out-of-fold quality metrics recorded when ``feature``'s imputer was trained."""
    empty = {"primary": None, "secondary": None, "mae": None}
    base = model_dir(feature)
    metrics_path = f"{base}/metrics/model_metrics.xlsx"
    if not os.path.exists(metrics_path):
        print(f"Warning: no metrics for {feature!r}; the feature will be dropped.")
        return empty

    try:
        metrics = pd.read_excel(metrics_path)
    except (ValueError, OSError) as exc:
        print(f"Warning: cannot read metrics for {feature!r} ({exc}); dropping it.")
        return empty

    if selection_type == "binary":
        return {
            "primary": _mean_metric(metrics, "OOF ROC AUC"),
            "secondary": _mean_metric(metrics, "OOF F1"),
            "mae": None,
        }
    if selection_type == "multiclass":
        return {
            "primary": _mean_metric(metrics, "OOF F1"),
            "secondary": _mean_metric(metrics, "OOF ROC AUC"),
            "mae": None,
        }
    return {
        "primary": _mean_metric(metrics, "OOF R2 Score"),
        "secondary": None,
        "mae": _mean_metric(metrics, "OOF MAE"),
    }


#: Per-type retention rule:
_RETENTION_RULES = {
    "binary": ("OOF AUROC", BINARY_AUROC_THRESHOLD, "OOF F1", BINARY_F1_THRESHOLD),
    "multiclass": ("OOF F1", MULTICLASS_F1_THRESHOLD, "OOF AUROC", MULTICLASS_AUROC_THRESHOLD),
    "regression": ("OOF R2", REGRESSION_R2_THRESHOLD, None, None),
}


def select_features_with_threshold(
    features, threshold=None, selection_type="binary", keep_maternal_history=False,
    verbose=False,
):
    """Split candidate cross-domain features into retained and removed.

    ``keep_maternal_history`` exempts the India-only ``PREV_PREGNANCY_FEATURES`` from
    the quality gate. It defaults to ``False`` because the manuscript reports those
    features as having FAILED the criteria (appendix Table S2a: AUROC 0.00-0.51) and
    having been excluded; exempting them made the default `scripts/03*` pipeline train
    a different feature space from the one every analysis script uses and from the
    one the Methods describe.
    """
    if selection_type not in _RETENTION_RULES:
        raise ValueError("selection_type must be binary, multiclass or regression")

    primary_name, primary_cut, secondary_name, secondary_cut = _RETENTION_RULES[selection_type]
    if threshold is not None:
        primary_cut = threshold

    selected, removed = [], []
    for feature in features:
        if feature in CONSTANT_ZERO_FEATURES:
            # Appendix Table S2: "Binary features with only one observed response,
            # such as smoking and some pregnancy complications were excluded."
            # They have no imputation model to gate, carry no signal, and are not
            # part of the harmonized feature set the manuscript describes, so they
            # are removed rather than silently added as a constant column.
            removed.append(feature)
            if verbose:
                print(f"{feature}: DROP   | constant in the source cohort, not imputable")
            continue
        if keep_maternal_history and feature in PREV_PREGNANCY_FEATURES:
            selected.append(feature)
            continue

        quality = _imputation_quality(feature, selection_type)
        primary, secondary = quality["primary"], quality["secondary"]

        keep = primary is not None and primary >= primary_cut
        if keep and secondary_cut is not None and secondary is not None:
            keep = secondary >= secondary_cut

        if verbose:
            parts = [
                f"{primary_name}="
                + ("n/a" if primary is None else f"{primary:.4f}")
                + f" (>= {primary_cut})"
            ]
            if secondary_name is not None:
                parts.append(
                    f"{secondary_name}="
                    + ("n/a" if secondary is None else f"{secondary:.4f}")
                    + f" (>= {secondary_cut})"
                )
            if quality["mae"] is not None:
                parts.append(f"MAE={quality['mae']:.4f}")
            print(f"{feature}: {'RETAIN' if keep else 'DROP  '} | " + " | ".join(parts))

        (selected if keep else removed).append(feature)

    return selected, removed
