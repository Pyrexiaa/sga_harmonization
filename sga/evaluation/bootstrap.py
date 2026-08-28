"""Percentile bootstrap confidence intervals, including cluster-aware resampling.

Scope. Per the Statistical Analysis section, the bootstrap is the interval method
for AUROC, AUPRC, ECE and Brier - the metrics that are not proportions. Every
proportion the manuscript reports (sensitivity, specificity, PPV, NPV, cohort-level
true- and false-positive rates) and every difference between two of them uses the
closed-form score intervals in ``sga.evaluation.proportions`` instead, because at
the Indian stratum's event counts a percentile bootstrap returns degenerate
intervals. ``bootstrap_rate_grid`` here produces bootstrap bounds for those rates
too, but only so the two methods can be reported side by side.

Where a pregnancy identifier is available, pass ``cluster_ids`` so whole
pregnancies rather than individual scans are resampled.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from sga.config import ALPHA, N_BOOTSTRAP, SEED
from sga.evaluation.metrics import expected_calibration_error

_METRIC_FUNCTIONS = {
    "auroc": lambda y, p: roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
    "auprc": lambda y, p: average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
    "ece": expected_calibration_error,
    "brier": brier_score_loss,
}


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=ALPHA, seed=SEED):
    """Percentile bootstrap CI of the MEAN of a list of per-fold values."""
    finite = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.RandomState(seed)
    means = [rng.choice(finite, size=len(finite), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(finite.mean()), float(lo), float(hi)


def bootstrap_statistic_ci(
    y_true,
    y_prob,
    statistic,
    n_boot=N_BOOTSTRAP,
    alpha=ALPHA,
    seed=SEED,
    cluster_ids=None,
):
    """Bootstrap CI of an arbitrary ``statistic(y_true, y_prob) -> float``.

    ``bootstrap_metric_ci`` covers the four named discrimination and calibration
    metrics; this is the general form used for threshold-dependent statistics such
    as the sensitivity and specificity swept across cut-offs in appendix Table S4.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.RandomState(seed)
    point = statistic(y_true, y_prob)

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        rows_by_cluster = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    replicates = []
    for _ in range(n_boot):
        if cluster_ids is not None:
            chosen = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            idx = np.concatenate([rows_by_cluster[c] for c in chosen])
        else:
            idx = rng.randint(0, len(y_true), len(y_true))
        try:
            value = statistic(y_true[idx], y_prob[idx])
        except ValueError:
            value = np.nan
        if not np.isnan(value):
            replicates.append(value)

    if not replicates:
        return float(point), float("nan"), float("nan")

    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def bootstrap_metric_ci(
    y_true,
    y_prob,
    metric="auroc",
    n_boot=N_BOOTSTRAP,
    alpha=ALPHA,
    seed=SEED,
    cluster_ids=None,
):
    """Bootstrap CI of a metric on one prediction set."""
    if metric not in _METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(_METRIC_FUNCTIONS)}")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    compute = _METRIC_FUNCTIONS[metric]
    rng = np.random.RandomState(seed)
    point = compute(y_true, y_prob)

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        rows_by_cluster = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    replicates = []
    for _ in range(n_boot):
        if cluster_ids is not None:
            chosen = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            idx = np.concatenate([rows_by_cluster[c] for c in chosen])
        else:
            idx = rng.randint(0, len(y_true), len(y_true))
        try:
            value = compute(y_true[idx], y_prob[idx])
        except ValueError:
            value = np.nan
        if not np.isnan(value):
            replicates.append(value)

    if not replicates:
        return float(point), float("nan"), float("nan")

    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def bootstrap_difference_ci(
    y_true, y_prob_a, y_prob_b, metric="auroc", n_boot=N_BOOTSTRAP, alpha=ALPHA, seed=SEED
):
    """Bootstrap CI of the paired difference ``metric(a) - metric(b)``."""
    compute = _METRIC_FUNCTIONS[metric]
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    rng = np.random.RandomState(seed)

    point = compute(y_true, y_prob_a) - compute(y_true, y_prob_b)
    replicates = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        try:
            value = compute(y_true[idx], y_prob_a[idx]) - compute(y_true[idx], y_prob_b[idx])
        except ValueError:
            continue
        if not np.isnan(value):
            replicates.append(value)

    if not replicates:
        return float(point), float("nan"), float("nan")

    lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def bootstrap_rate_grid(
    y_true,
    y_prob,
    thresholds,
    rates=("sensitivity", "specificity"),
    n_boot=N_BOOTSTRAP,
    alpha=ALPHA,
    seed=SEED,
    cluster_ids=None,
):
    """Bootstrap CI BOUNDS for threshold-dependent rates across a whole grid.

    Two reasons this exists rather than calling ``bootstrap_statistic_ci`` once per
    (threshold, rate) cell:

    * Consistency. One resample is drawn per iteration and reused for every
      threshold and every rate, so the bounds down a column of appendix Table S4
      move together instead of each being a separate random experiment.
    * Cost. The naive form redraws ``n_boot`` resamples for every cell - 17
      thresholds x 2 rates x 3 cohorts x 2,000 iterations, each rebuilding a
      confusion matrix. Here every replicate is reduced to a vector of row
      multiplicities and the whole grid is evaluated with four matrix products.

    Only the interval is returned. The point estimate stays with the caller,
    because the reported rate is the observed proportion (and its interval, per the
    Statistical Analysis section, is Wilson's) - not the mean of a resampling
    distribution.

    Returns ``{rate: [(low, high), ...]}`` in the order of ``thresholds``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    n = len(y_true)
    nan_grid = [(float("nan"), float("nan"))] * len(thresholds)
    if n == 0:
        return {rate: list(nan_grid) for rate in rates}

    rng = np.random.RandomState(seed)
    # predicted[t, i] is True when row i is called positive at threshold t.
    predicted = y_prob[None, :] >= thresholds[:, None]
    positive = y_true.astype(float)
    negative = 1.0 - positive

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        rows_by_cluster = [np.where(cluster_ids == c)[0] for c in unique_clusters]

    multiplicities = np.empty((n, int(n_boot)), dtype=float)
    for replicate in range(int(n_boot)):
        if cluster_ids is None:
            idx = rng.randint(0, n, n)
        else:
            chosen = rng.randint(0, len(unique_clusters), len(unique_clusters))
            idx = np.concatenate([rows_by_cluster[c] for c in chosen])
        multiplicities[:, replicate] = np.bincount(idx, minlength=n)

    # Each product is (thresholds x rows) @ (rows x replicates).
    weighted_positive = multiplicities * positive[:, None]
    weighted_negative = multiplicities * negative[:, None]
    true_positive = predicted @ weighted_positive
    false_negative = (~predicted) @ weighted_positive
    false_positive = predicted @ weighted_negative
    true_negative = (~predicted) @ weighted_negative

    numerators = {
        "sensitivity": true_positive,
        "specificity": true_negative,
        "ppv": true_positive,
        "npv": true_negative,
    }
    denominators = {
        "sensitivity": true_positive + false_negative,
        "specificity": true_negative + false_positive,
        "ppv": true_positive + false_positive,
        "npv": true_negative + false_negative,
    }

    low_q, high_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    results = {}
    for rate in rates:
        if rate not in numerators:
            raise ValueError(f"Unknown rate {rate!r}; choose from {sorted(numerators)}")
        with np.errstate(invalid="ignore", divide="ignore"):
            replicates = np.where(
                denominators[rate] > 0, numerators[rate] / denominators[rate], np.nan
            )
        bounds = []
        for index in range(len(thresholds)):
            values = replicates[index]
            values = values[~np.isnan(values)]
            if values.size == 0:
                bounds.append((float("nan"), float("nan")))
                continue
            low, high = np.percentile(values, [low_q, high_q])
            bounds.append((float(low), float(high)))
        results[rate] = bounds
    return results
