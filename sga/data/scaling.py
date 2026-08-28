"""Standardisation helpers."""

from __future__ import annotations

import numpy as np

_METHODS = ("std", "minmax")


def _check(method: str) -> None:
    if method not in _METHODS:
        raise ValueError(f"Unknown scaling method {method!r}; choose from {_METHODS}")


def scale_feature_train(df, method="std"):
    """Fit and apply scaling on the training partition."""
    _check(method)
    scaled = df.copy()
    std_or_min, mean_or_max = {}, {}

    for col in df.columns:
        if method == "std":
            std, mean = np.std(df[col]), np.mean(df[col])
            scaled[col] = (df[col] - mean) / std
            std_or_min[col], mean_or_max[col] = std, mean
        else:
            lo, hi = np.min(df[col]), np.max(df[col])
            scaled[col] = (df[col] - lo) / (hi - lo)
            std_or_min[col], mean_or_max[col] = lo, hi

    return scaled, std_or_min, mean_or_max


def scale_feature_test(df, training_std_or_min, training_mean_or_max, method="std"):
    """Apply training-fitted scaling statistics to held-out data."""
    _check(method)
    scaled = df.copy()
    for col in df.columns:
        if method == "std":
            scaled[col] = (df[col] - training_mean_or_max[col]) / training_std_or_min[col]
        else:
            span = training_mean_or_max[col] - training_std_or_min[col]
            scaled[col] = (df[col] - training_std_or_min[col]) / span
    return scaled


def descale_feature(df, training_std_or_min, training_mean_or_max, method="std"):
    """Invert :func:`scale_feature_test` so saved CSVs hold clinical units."""
    _check(method)
    descaled = df.copy()
    for col in df.columns:
        if method == "std":
            descaled[col] = df[col] * training_std_or_min[col] + training_mean_or_max[col]
        else:
            span = training_mean_or_max[col] - training_std_or_min[col]
            descaled[col] = df[col] * span + training_std_or_min[col]
    return descaled


def scale_single_value(value, col, training_std_or_min, training_mean_or_max, method="std"):
    """Scale one scalar, used by the deployment/inference path."""
    _check(method)
    if method == "std":
        return (value - training_mean_or_max[col]) / training_std_or_min[col]
    span = training_mean_or_max[col] - training_std_or_min[col]
    return (value - training_std_or_min[col]) / span
