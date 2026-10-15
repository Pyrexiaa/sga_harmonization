"""Platt scaling fitted without touching the reported test labels."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from sga.config import ECE_BINS, SEED

_EPS = 1e-6


def logit(p):
    """Numerically safe logit transform of a probability vector."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_platt(y_true, p_raw):
    """Fit a Platt scaler (logistic regression on the logit of ``p_raw``)."""
    return LogisticRegression().fit(logit(p_raw).reshape(-1, 1), np.asarray(y_true, dtype=int))


def apply_platt(calibrator, p_raw):
    """Apply a fitted Platt scaler."""
    return calibrator.predict_proba(logit(p_raw).reshape(-1, 1))[:, 1]


def platt_cross_fitted(y_true, p_raw, n_splits=5, seed=SEED):
    """Leakage-free calibrated probabilities via cross-fitted Platt scaling."""
    y_true = np.asarray(y_true, dtype=int)
    p_raw = np.asarray(p_raw, dtype=float)
    if len(np.unique(y_true)) < 2:
        return p_raw.copy()

    n_splits = int(min(n_splits, np.bincount(y_true).min()))
    if n_splits < 2:
        return apply_platt(fit_platt(y_true, p_raw), p_raw)

    X = logit(p_raw).reshape(-1, 1)
    p_cal = np.full_like(p_raw, np.nan)
    for train_idx, test_idx in StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    ).split(X, y_true):
        calibrator = LogisticRegression().fit(X[train_idx], y_true[train_idx])
        p_cal[test_idx] = calibrator.predict_proba(X[test_idx])[:, 1]

    unfilled = np.isnan(p_cal)
    p_cal[unfilled] = p_raw[unfilled]
    return p_cal


def reliability_curve(y_true, y_prob, n_bins=ECE_BINS, min_count=1):
    """Bin centres, observed frequency and mean predicted probability per bin."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)

    centres, observed, predicted, counts = [], [], [], []
    for i in range(n_bins):
        upper_inclusive = i == n_bins - 1
        in_bin = (y_prob >= edges[i]) & (
            y_prob <= edges[i + 1] if upper_inclusive else y_prob < edges[i + 1]
        )
        if in_bin.sum() < min_count:
            continue
        centres.append((edges[i] + edges[i + 1]) / 2)
        observed.append(y_true[in_bin].mean())
        predicted.append(y_prob[in_bin].mean())
        counts.append(int(in_bin.sum()))

    return np.array(centres), np.array(observed), np.array(predicted), np.array(counts)
