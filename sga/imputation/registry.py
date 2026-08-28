"""Locations of the pretrained per-feature cross-domain imputation models."""

from __future__ import annotations

from pathlib import Path

from sga.config import (
    IMPUTER_DIR,
    INDIA_BINARY_FEATURES,
    INDIA_REGRESSION_FEATURES,
    MALAYSIA_MULTICLASS_FEATURES,
    MALAYSIA_REGRESSION_FEATURES,
)

# Malaysia-native features imputed into India.
_MALAYSIA_SOURCE = sorted(
    set(MALAYSIA_MULTICLASS_FEATURES + MALAYSIA_REGRESSION_FEATURES + ["bpd", "cpr", "ute_api"])
)
# India-native features imputed into Malaysia.
_INDIA_SOURCE = sorted(set(INDIA_BINARY_FEATURES + INDIA_REGRESSION_FEATURES))

#: Feature -> the imputer sub-directory name, independent of where the weights live.
IMPUTER_SUBDIRS = {
    **{f: f"train_malaysia_predict_{f}" for f in _MALAYSIA_SOURCE},
    **{f: f"train_india_predict_{f}" for f in _INDIA_SOURCE},
}

#: Kept for backwards compatibility; resolved against the default imputer directory.
IMPUTATION_MODELS = {
    feature: str(IMPUTER_DIR / subdir) for feature, subdir in IMPUTER_SUBDIRS.items()
}


def model_dir(feature: str, base_dir=None) -> str:
    """Directory of the pretrained imputer for ``feature``.

    ``base_dir`` defaults to ``config.IMPUTER_DIR``. It is resolved on every call, so
    a caller that trained its imputers somewhere else can point the quality gate at
    the same place instead of silently finding no metrics.
    """
    try:
        subdir = IMPUTER_SUBDIRS[feature]
    except KeyError as exc:
        raise KeyError(
            f"No imputation model registered for {feature!r}. "
            f"Known features: {sorted(IMPUTER_SUBDIRS)}"
        ) from exc
    return str(Path(base_dir or IMPUTER_DIR) / subdir)
