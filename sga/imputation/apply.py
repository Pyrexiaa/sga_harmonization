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


def load_persisted_imputer(feature, kind, fold=0):
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
            model = load_persisted_imputer(feature, kind)

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


def _multiclass_gate(base):
    """Majority-baseline F1 and macro AUROC interval written by the imputer trainer."""
    from sga.imputation.train_imputers import MULTICLASS_GATE_FILE

    path = os.path.join(base, "metrics", MULTICLASS_GATE_FILE)
    if not os.path.exists(path):
        return None
    row = pd.read_csv(path).iloc[0]
    return {
        "baseline_f1": float(row["majority_baseline_f1"]),
        "auroc_ci": (float(row["macro_ovr_auroc_ci_low"]), float(row["macro_ovr_auroc_ci_high"])),
    }


def _imputation_quality(feature, selection_type):
    """Out-of-fold quality metrics recorded when ``feature``'s imputer was trained."""
    empty = {"primary": None, "secondary": None, "mae": None, "gate": None}
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
            "gate": None,
        }
    if selection_type == "multiclass":
        return {
            "primary": _mean_metric(metrics, "OOF F1"),
            "secondary": _mean_metric(metrics, "OOF ROC AUC"),
            "mae": None,
            "gate": _multiclass_gate(base),
        }
    return {
        "primary": _mean_metric(metrics, "OOF R2 Score"),
        "secondary": None,
        "mae": _mean_metric(metrics, "OOF MAE"),
        "gate": None,
    }


_RETENTION_RULES = {
    "binary": ("OOF AUROC", BINARY_AUROC_THRESHOLD, "OOF F1", BINARY_F1_THRESHOLD),
    "multiclass": ("OOF F1", MULTICLASS_F1_THRESHOLD, "OOF AUROC", MULTICLASS_AUROC_THRESHOLD),
    "regression": ("OOF R2", REGRESSION_R2_THRESHOLD, None, None),
}


def passes_retention_rule(selection_type, primary, secondary=None, gate=None, primary_cut=None):
    """Apply the per-type retention criteria of appendix Table S2 (all bounds strict)."""
    _, default_primary, _, secondary_cut = _RETENTION_RULES[selection_type]
    primary_cut = default_primary if primary_cut is None else primary_cut
    if primary is None or not primary > primary_cut:
        return False
    if secondary_cut is not None and (secondary is None or not secondary > secondary_cut):
        return False
    if selection_type == "multiclass":
        if gate is None:
            return False
        low, high = gate["auroc_ci"]
        if not primary > gate["baseline_f1"]:
            return False
        if not (low > 0.5 or high < 0.5):
            return False
    return True


def select_features_with_threshold(
    features, threshold=None, selection_type="binary", keep_maternal_history=False,
    verbose=False,
):
    """Split candidate cross-domain features into retained and removed.

    ``keep_maternal_history`` exempts the India-only ``PREV_PREGNANCY_FEATURES`` from
    the quality gate. It defaults to ``False`` because those features failed the
    criteria (appendix Table S2a) and were excluded.
    """
    if selection_type not in _RETENTION_RULES:
        raise ValueError("selection_type must be binary, multiclass or regression")

    primary_name, primary_cut, secondary_name, secondary_cut = _RETENTION_RULES[selection_type]
    if threshold is not None:
        primary_cut = threshold

    selected, removed = [], []
    for feature in features:
        if feature in CONSTANT_ZERO_FEATURES:
            removed.append(feature)
            if verbose:
                print(f"{feature}: DROP   | constant in the source cohort, not imputable")
            continue
        if keep_maternal_history and feature in PREV_PREGNANCY_FEATURES:
            selected.append(feature)
            continue

        quality = _imputation_quality(feature, selection_type)
        primary, secondary = quality["primary"], quality["secondary"]
        keep = passes_retention_rule(
            selection_type, primary, secondary, quality["gate"], primary_cut
        )

        if verbose:
            parts = [
                f"{primary_name}="
                + ("n/a" if primary is None else f"{primary:.4f}")
                + f" (> {primary_cut})"
            ]
            if secondary_name is not None:
                parts.append(
                    f"{secondary_name}="
                    + ("n/a" if secondary is None else f"{secondary:.4f}")
                    + f" (> {secondary_cut})"
                )
            if quality["gate"] is not None:
                low, high = quality["gate"]["auroc_ci"]
                parts.append(
                    f"majority-baseline F1={quality['gate']['baseline_f1']:.4f}, "
                    f"AUROC CI={low:.3f}-{high:.3f}"
                )
            if quality["mae"] is not None:
                parts.append(f"MAE={quality['mae']:.4f}")
            print(f"{feature}: {'RETAIN' if keep else 'DROP  '} | " + " | ".join(parts))

        (selected if keep else removed).append(feature)

    return selected, removed
