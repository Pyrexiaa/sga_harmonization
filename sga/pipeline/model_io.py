"""Loading, scoring and persisting already-trained fold-models."""

from __future__ import annotations

import glob
import os
import pickle

import numpy as np
import pandas as pd
import torch

from sga.config import DECISION_THRESHOLD, N_FOLDS_CV
from sga.models.architecture import MODEL_SIZES
from sga.models.torch_utils import DEVICE
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA

CATBOOST, SKLEARN, DNN = "catboost", "ml", "dnn"

# ``sklearn`` is accepted as a synonym of ``ml`` because the round-1 scripts used both
# spellings for the same weight layout.
_FAMILY_ALIASES = {
    CATBOOST: CATBOOST,
    SKLEARN: SKLEARN,
    "sklearn": SKLEARN,
    DNN: DNN,
}

# Candidate file suffixes, in the order the original scripts probed them.
_WEIGHT_SUFFIXES = {
    CATBOOST: ("", ".cbm"),
    SKLEARN: (".pkl", ".joblib", ""),
    DNN: (".pth", ".pt"),
}

WEIGHTS_SUBDIR = "model_weights"


def _resolve_family(family):
    """Normalise a family name."""
    try:
        return _FAMILY_ALIASES[family]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model family {family!r}; choose from {sorted(set(_FAMILY_ALIASES))}"
        ) from exc


def _weights_dir(download_path, weights_subdir=WEIGHTS_SUBDIR):
    """Directory holding the per-fold weight files of a training run."""
    return os.path.join(download_path, weights_subdir) if weights_subdir else str(download_path)


def weights_exist(download_path, n_folds=N_FOLDS_CV, weights_subdir=WEIGHTS_SUBDIR):
    """Check that a weight file exists for every fold of a training run."""
    directory = _weights_dir(download_path, weights_subdir)
    if not os.path.isdir(directory):
        return False
    return all(
        glob.glob(os.path.join(directory, f"model_{fold}*")) for fold in range(n_folds)
    )


def _weight_path(family, download_path, fold, weights_subdir=WEIGHTS_SUBDIR):
    """First existing weight file for ``fold``, or None."""
    directory = _weights_dir(download_path, weights_subdir)
    for suffix in _WEIGHT_SUFFIXES[family]:
        candidate = os.path.join(directory, f"model_{fold}{suffix}")
        if os.path.exists(candidate) and os.path.isfile(candidate):
            return candidate
    return None


def load_trained_model(
    family,
    download_path,
    fold,
    n_features=None,
    dnn_config=None,
    model_size="large",
    weights_subdir=WEIGHTS_SUBDIR,
):
    """Rebuild one saved fold-model from disk."""
    family = _resolve_family(family)
    path = _weight_path(family, download_path, fold, weights_subdir)
    if path is None:
        return None

    if family == CATBOOST:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(path)
        return model

    if family == SKLEARN:
        if path.endswith(".joblib"):
            import joblib

            return joblib.load(path)
        with open(path, "rb") as handle:
            return pickle.load(handle)

    if n_features is None or dnn_config is None:
        raise ValueError("Loading a DNN requires both n_features and dnn_config")
    dropout_rate, layer_output_size = dnn_config[0], dnn_config[1]
    net = MODEL_SIZES[model_size](n_features, dropout_rate, layer_output_size)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.to(DEVICE)
    net.eval()
    return net


def predict_proba(model, family, X):
    """Score a scaled design matrix and return P(SGA)."""
    family = _resolve_family(family)
    if family in (CATBOOST, SKLEARN):
        return np.asarray(model.predict_proba(X))[:, 1]

    values = X.values if hasattr(X, "values") else np.asarray(X)
    with torch.no_grad():
        logits = (
            model(torch.as_tensor(values, dtype=torch.float32, device=DEVICE))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    return 1.0 / (1.0 + np.exp(-logits))


def save_predictions(out_path, feature_df, y_true, y_pred, y_prob, country_arr=None):
    """Write one prediction CSV (features plus label, prediction and probability)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = pd.DataFrame(feature_df).copy()
    out["Actual"] = y_true
    out["Prediction"] = y_pred
    out["predicted_probability"] = y_prob
    if country_arr is not None:
        out["country"] = [
            "malaysia" if c == MALAYSIA else "india" for c in np.asarray(country_arr)
        ]
    out.to_csv(out_path, index=False)
    print(f"    saved {out_path}  ({len(out)} rows)")
    return out_path


def predict_labels_and_proba(model, family, X, threshold=DECISION_THRESHOLD):
    """Convenience wrapper returning ``(y_pred, y_prob)`` at ``threshold``."""
    y_prob = predict_proba(model, family, X)
    return (y_prob >= threshold).astype(int), y_prob


__all__ = [
    "CATBOOST",
    "SKLEARN",
    "DNN",
    "WEIGHTS_SUBDIR",
    "INDIA",
    "MALAYSIA",
    "load_trained_model",
    "predict_proba",
    "predict_labels_and_proba",
    "save_predictions",
    "weights_exist",
]
