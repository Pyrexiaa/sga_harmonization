"""Cohort-level performance-disparity metrics (country as the grouping variable).

This is the machinery behind the manuscript's "Cohort-Level Fairness Analysis"
section and appendix Table S5. Three decisions here are deliberate and each is
required by the text, so changing one silently invalidates a reported number.

1. Metrics are computed on the CALIBRATED probabilities of the external test fold.
   Thresholding raw and Platt-scaled scores at the same cut-off does not select the
   same patients, so the cohort rates differ between the two.

2. Differences are SIGNED (Malaysia minus India). Signed intervals can contain 0,
   which lets "is there a gap at all" be a real question; an interval built around
   an absolute difference is biased away from zero and can never cover it. The
   conventional absolute value is reported alongside under ``*_abs``.

3. Intervals are Wilson (rates) and Newcombe hybrid score (differences), per the
   Statistical Analysis section, NOT the percentile bootstrap. With 15 Indian SGA
   events a resampling interval is unreliable and frequently degenerate; bootstrap
   intervals are still produced by ``bootstrap_fairness_sweep`` and reported
   alongside, but they are not the primary numbers.

Equalized odds is defined on absolute discrepancies, so its point estimate is
``max(|dTPR|, |dFPR|)``; the interval reported with it is the SIGNED interval of
whichever component attained that maximum, so the direction stays recoverable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sga.config import (
    ALPHA,
    DECISION_THRESHOLD,
    INDIA,
    MALAYSIA,
    MIN_POSITIVE_PREDICTIONS_FOR_PPV,
    N_BOOTSTRAP,
    SEED,
    THRESHOLD_GRID,
)
from sga.evaluation.metrics import confusion_counts
from sga.evaluation.proportions import (
    NOT_ESTIMABLE,
    difference_with_ci,
    percentile_ci,
    rate_with_ci,
)

#: Cohort name -> the code used in ``country_arr``. Malaysia first, matching the
#: Malaysia-first row order every fold builder produces.
GROUPS = (("malaysia", MALAYSIA), ("india", INDIA))

#: Rate name -> (numerator counts, denominator counts) within one cohort.
_COHORT_RATES = {
    "tpr": (("tp",), ("tp", "fn")),          # sensitivity
    "fpr": (("fp",), ("fp", "tn")),
    "tnr": (("tn",), ("tn", "fp")),          # specificity
    "ppv": (("tp",), ("tp", "fp")),
    "npv": (("tn",), ("tn", "fn")),
    "pred_rate": (("tp", "fp"), ("tp", "fp", "fn", "tn")),
    "prevalence": (("tp", "fn"), ("tp", "fp", "fn", "tn")),
}

#: Difference name -> the cohort rate it is the Malaysia-minus-India gap of.
_DIFFERENCES = {
    "equal_opportunity_diff": "tpr",
    "fpr_diff": "fpr",
    "demographic_parity_diff": "pred_rate",
    "predictive_parity_diff": "ppv",
}

DIFFERENCE_NAMES = tuple(_DIFFERENCES) + ("equalized_odds_diff",)


# ── Per-cohort counts and rates ──────────────────────────────────────────────


def cohort_counts(y_true, y_pred, mask):
    """Confusion counts plus n, event count and the cohort's mask size."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.asarray(mask, dtype=bool)
    counts = confusion_counts(y_true[mask], y_pred[mask])
    counts["n"] = int(mask.sum())
    counts["n_pos"] = int(y_true[mask].sum()) if mask.sum() else 0
    return counts


def _rate_fraction(counts, rate):
    """``(numerator, denominator)`` of one cohort rate, from its counts."""
    numerator_keys, denominator_keys = _COHORT_RATES[rate]
    return (
        sum(counts[k] for k in numerator_keys),
        sum(counts[k] for k in denominator_keys),
    )


def cohort_rates(counts, alpha=ALPHA):
    """Every cohort rate as a Wilson :class:`Interval`, keyed by rate name."""
    return {
        rate: rate_with_ci(*_rate_fraction(counts, rate), alpha=alpha)
        for rate in _COHORT_RATES
    }


# ── One threshold ────────────────────────────────────────────────────────────


def fairness_points(y_true, y_prob, country_arr, threshold=DECISION_THRESHOLD):
    """Signed difference POINT estimates only - no intervals, no formatting.

    The bootstrap evaluates this tens of thousands of times, so it deliberately
    skips the Wilson and Newcombe computation that
    :func:`fairness_at_threshold` performs.

    ``equalized_odds_diff`` is returned SIGNED here (the component with the larger
    magnitude), because a resampling distribution of an absolute value cannot be
    used to test "is there a gap at all".
    """
    y_true = np.asarray(y_true)
    country_arr = np.asarray(country_arr)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)

    counts = {
        name: cohort_counts(y_true, y_pred, country_arr == code) for name, code in GROUPS
    }
    points = {}
    for difference, rate in _DIFFERENCES.items():
        malaysia_numerator, malaysia_denominator = _rate_fraction(counts["malaysia"], rate)
        india_numerator, india_denominator = _rate_fraction(counts["india"], rate)
        if malaysia_denominator == 0 or india_denominator == 0:
            points[difference] = float("nan")
        else:
            points[difference] = (
                malaysia_numerator / malaysia_denominator
                - india_numerator / india_denominator
            )

    tpr_gap, fpr_gap = points["equal_opportunity_diff"], points["fpr_diff"]
    if np.isnan(tpr_gap) and np.isnan(fpr_gap):
        points["equalized_odds_diff"] = float("nan")
    elif np.isnan(fpr_gap) or (not np.isnan(tpr_gap) and abs(tpr_gap) >= abs(fpr_gap)):
        points["equalized_odds_diff"] = tpr_gap
    else:
        points["equalized_odds_diff"] = fpr_gap
    return points


def fairness_at_threshold(
    y_true,
    y_prob,
    country_arr,
    threshold=DECISION_THRESHOLD,
    alpha=ALPHA,
    min_positive_predictions=MIN_POSITIVE_PREDICTIONS_FOR_PPV,
):
    """Every cohort-level fairness quantity at one cut-off, as a flat dict.

    ``y_prob`` must be the CALIBRATED probability, and ``country_arr`` the cohort
    code per row (see :data:`GROUPS`).

    Returned keys, per cohort ``c`` in ``{malaysia, india}``:

    * ``n_c``, ``n_pos_c``, ``tp_c``, ``fp_c``, ``fn_c``, ``tn_c``
    * ``{tpr,fpr,tnr,ppv,npv,pred_rate,prevalence}_c`` with ``_ci_low``,
      ``_ci_high`` and a formatted ``_str`` (Wilson)

    and, for each signed difference in :data:`DIFFERENCE_NAMES`, the point estimate
    with its Newcombe ``_ci_low`` / ``_ci_high`` / ``_str``, its absolute value
    under ``_abs``, and ``_ci_excludes_0``.

    ``predictive_parity_diff_flag`` is ``"N/A"`` when a cohort predicted no
    positives at all and ``"*"`` when the Indian cohort has fewer than
    ``min_positive_predictions`` of them - the two footnotes of Table S5.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    country_arr = np.asarray(country_arr)
    y_pred = (y_prob >= threshold).astype(int)

    row = {"threshold": float(threshold)}
    counts = {}
    for name, code in GROUPS:
        counts[name] = cohort_counts(y_true, y_pred, country_arr == code)
        for key, value in counts[name].items():
            row[f"{key}_{name}"] = value
        for rate, interval in cohort_rates(counts[name], alpha).items():
            row.update(interval.as_dict(f"{rate}_{name}"))

    malaysia, india = counts["malaysia"], counts["india"]
    intervals = {}
    for difference, rate in _DIFFERENCES.items():
        malaysia_fraction = _rate_fraction(malaysia, rate)
        india_fraction = _rate_fraction(india, rate)
        intervals[difference] = difference_with_ci(
            *malaysia_fraction, *india_fraction, alpha=alpha
        )

    # Equalized odds: the absolute maximum of the two error-rate gaps, carrying the
    # signed interval of whichever component attained it.
    tpr_gap, fpr_gap = intervals["equal_opportunity_diff"], intervals["fpr_diff"]
    if not (tpr_gap.is_estimable and fpr_gap.is_estimable):
        dominant = tpr_gap if tpr_gap.is_estimable else fpr_gap
        equalized = dominant if dominant.is_estimable else NOT_ESTIMABLE
        dominant_component = "tpr" if dominant is tpr_gap else "fpr"
    elif abs(tpr_gap.point) >= abs(fpr_gap.point):
        equalized, dominant_component = tpr_gap, "tpr"
    else:
        equalized, dominant_component = fpr_gap, "fpr"
    intervals["equalized_odds_diff"] = equalized
    row["equalized_odds_dominant_component"] = dominant_component

    for name in DIFFERENCE_NAMES:
        interval = intervals[name]
        row.update(interval.as_dict(name))
        row[f"{name}_abs"] = abs(interval.point) if interval.is_estimable else float("nan")
        row[f"{name}_ci_excludes_0"] = interval.excludes(0.0)

    # Equalized odds is conventionally quoted as an absolute discrepancy; the point
    # estimate follows that convention while the interval stays signed. The signed
    # value is kept too, because that is what the bootstrap resamples.
    row["equalized_odds_diff_signed"] = intervals["equalized_odds_diff"].point
    row["equalized_odds_diff"] = row["equalized_odds_diff_abs"]

    india_positive_predictions = india["tp"] + india["fp"]
    malaysia_positive_predictions = malaysia["tp"] + malaysia["fp"]
    if min(india_positive_predictions, malaysia_positive_predictions) == 0:
        row["predictive_parity_diff_flag"] = "N/A"
    elif india_positive_predictions < min_positive_predictions:
        row["predictive_parity_diff_flag"] = "*"
    else:
        row["predictive_parity_diff_flag"] = ""
    row["n_positive_predictions_india"] = int(india_positive_predictions)
    row["n_positive_predictions_malaysia"] = int(malaysia_positive_predictions)
    return row


def fairness_threshold_sweep(
    y_true,
    y_prob,
    country_arr,
    thresholds=THRESHOLD_GRID,
    alpha=ALPHA,
    min_positive_predictions=MIN_POSITIVE_PREDICTIONS_FOR_PPV,
):
    """:func:`fairness_at_threshold` across a grid - appendix Table S5."""
    return pd.DataFrame(
        [
            fairness_at_threshold(
                y_true,
                y_prob,
                country_arr,
                threshold,
                alpha=alpha,
                min_positive_predictions=min_positive_predictions,
            )
            for threshold in thresholds
        ]
    )


# ── Bootstrap intervals, reported alongside the score intervals ──────────────


def _resample_indices(rng, country_arr, cluster_ids, stratify_by_cohort):
    """One bootstrap resample: cohort-stratified and/or pregnancy-clustered.

    Stratifying keeps ``n_malaysia`` and ``n_india`` fixed in every replicate. With
    46 Indian rows an unstratified resample regularly draws a replicate containing
    no Indian SGA events at all, which silently biases the interval.
    """
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)

    if not stratify_by_cohort:
        if cluster_ids is None:
            return rng.randint(0, len(country_arr), len(country_arr))
        unique = np.unique(cluster_ids)
        rows_by_cluster = {c: np.where(cluster_ids == c)[0] for c in unique}
        chosen = rng.choice(unique, size=len(unique), replace=True)
        return np.concatenate([rows_by_cluster[c] for c in chosen])

    parts = []
    for _, code in GROUPS:
        rows = np.where(country_arr == code)[0]
        if len(rows) == 0:
            continue
        if cluster_ids is None:
            parts.append(rng.choice(rows, size=len(rows), replace=True))
            continue
        cohort_clusters = cluster_ids[rows]
        unique = np.unique(cohort_clusters)
        rows_by_cluster = {c: rows[cohort_clusters == c] for c in unique}
        chosen = rng.choice(unique, size=len(unique), replace=True)
        parts.append(np.concatenate([rows_by_cluster[c] for c in chosen]))
    return np.concatenate(parts) if parts else np.arange(len(country_arr))


def bootstrap_fairness_sweep(
    y_true,
    y_prob,
    country_arr,
    thresholds=THRESHOLD_GRID,
    cluster_ids=None,
    n_boot=N_BOOTSTRAP,
    alpha=ALPHA,
    seed=SEED,
    stratify_by_cohort=True,
):
    """Percentile-bootstrap intervals for the signed differences, for comparison.

    One resample is drawn per iteration and reused for every threshold and every
    metric, so the intervals are mutually consistent: a replicate that happens to
    contain few Indian SGA events moves all of them the same way.

    The calibrated probabilities are treated as fixed. Re-fitting Platt scaling
    inside each replicate would also propagate calibration uncertainty, but the
    manuscript reports the calibrated probability as a fixed transformation of the
    model output, and this matches that.

    Returns a DataFrame with ``threshold`` and, per difference,
    ``<name>_boot_ci_low`` / ``_boot_ci_high`` / ``_boot_p``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    country_arr = np.asarray(country_arr)
    rng = np.random.RandomState(seed)

    replicates = {t: {name: [] for name in DIFFERENCE_NAMES} for t in thresholds}
    for _ in range(int(n_boot)):
        idx = _resample_indices(rng, country_arr, cluster_ids, stratify_by_cohort)
        resampled_y = y_true[idx]
        resampled_p = y_prob[idx]
        resampled_country = country_arr[idx]
        for threshold in thresholds:
            points = fairness_points(
                resampled_y, resampled_p, resampled_country, threshold
            )
            for name in DIFFERENCE_NAMES:
                replicates[threshold][name].append(points[name])

    rows = []
    for threshold in thresholds:
        row = {"threshold": float(threshold), "n_boot": int(n_boot)}
        for name in DIFFERENCE_NAMES:
            values = np.asarray(replicates[threshold][name], dtype=float)
            values = values[~np.isnan(values)]
            low, high = percentile_ci(values, alpha)
            row[f"{name}_boot_ci_low"] = low
            row[f"{name}_boot_ci_high"] = high
            if values.size:
                below = float((values <= 0).mean())
                above = float((values >= 0).mean())
                row[f"{name}_boot_p"] = min(1.0, 2 * min(below, above))
            else:
                row[f"{name}_boot_p"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ── Backwards-compatible absolute-gap helpers ────────────────────────────────


def compute_group_rates(y_true, y_pred, group_mask):
    """TPR, FPR, PPV, predicted-positive rate and n within one group.

    Undefined rates are NaN rather than 0, so an empty denominator stays
    distinguishable from a cohort that genuinely never fires.
    """
    counts = cohort_counts(y_true, y_pred, group_mask)
    if counts["n"] == 0:
        return {"tpr": np.nan, "fpr": np.nan, "ppv": np.nan, "pred_rate": np.nan, "n": 0}

    rates = {}
    for rate in ("tpr", "fpr", "ppv", "pred_rate"):
        numerator, denominator = _rate_fraction(counts, rate)
        rates[rate] = numerator / denominator if denominator else np.nan
    rates["n"] = counts["n"]
    return rates


def compute_fairness_metrics(y_true, y_pred, country_arr):
    """Absolute equal-opportunity, equalized-odds, demographic- and predictive-parity gaps.

    Kept for the per-fold supporting tables that predate the score intervals. The
    manuscript's reported values come from :func:`fairness_at_threshold`, which is
    signed and interval-bearing; this helper carries no uncertainty.
    """
    country_arr = np.asarray(country_arr)
    malaysia = compute_group_rates(y_true, y_pred, country_arr == MALAYSIA)
    india = compute_group_rates(y_true, y_pred, country_arr == INDIA)

    tpr_gap = abs(malaysia["tpr"] - india["tpr"])
    fpr_gap = abs(malaysia["fpr"] - india["fpr"])
    equalized_odds = (
        max(tpr_gap, fpr_gap) if not (np.isnan(tpr_gap) or np.isnan(fpr_gap)) else np.nan
    )

    return {
        "eod": tpr_gap,                                          # equal opportunity
        "equalized_odds_diff": equalized_odds,
        "dpd": abs(malaysia["pred_rate"] - india["pred_rate"]),  # demographic parity
        "predictive_parity": abs(malaysia["ppv"] - india["ppv"]),
        "tpr_malaysia": malaysia["tpr"],
        "tpr_india": india["tpr"],
        "fpr_malaysia": malaysia["fpr"],
        "fpr_india": india["fpr"],
        "n_malaysia": malaysia["n"],
        "n_india": india["n"],
    }


__all__ = [
    "DIFFERENCE_NAMES",
    "GROUPS",
    "bootstrap_fairness_sweep",
    "cohort_counts",
    "cohort_rates",
    "compute_fairness_metrics",
    "compute_group_rates",
    "fairness_at_threshold",
    "fairness_points",
    "fairness_threshold_sweep",
]
