"""Cohort-level fairness: signed gaps, score intervals and the equalized-odds rule.

Each scenario fixes both cohorts' confusion matrices (Malaysia 807 scans / 169 SGA,
India 46 scans / 15 SGA), so every rate and gap has a closed-form reference.
"""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

from sga.config import INDIA, MALAYSIA
from sga.evaluation.fairness import (
    bootstrap_fairness_sweep,
    fairness_at_threshold,
    fairness_points,
    fairness_threshold_sweep,
)

TOLERANCE = 5e-5
THRESHOLD = 0.5
_HIGH, _LOW = 0.9, 0.1  # scores either side of THRESHOLD


def cohort_arrays(counts, country_code):
    """Build (y, p, country) reproducing one cohort's ``(tp, fp, fn, tn)``."""
    tp, fp, fn, tn = counts
    y = np.array([1] * tp + [0] * fp + [1] * fn + [0] * tn, dtype=int)
    p = np.array([_HIGH] * (tp + fp) + [_LOW] * (fn + tn), dtype=float)
    country = np.full(len(y), country_code, dtype=int)
    return y, p, country


def fold(malaysia_counts, india_counts):
    """Malaysia-first test block reproducing both cohorts' confusion matrices."""
    parts = [
        cohort_arrays(malaysia_counts, MALAYSIA),
        cohort_arrays(india_counts, INDIA),
    ]
    return tuple(np.concatenate(arrays) for arrays in zip(*parts))


#: name -> ((tp, fp, fn, tn) Malaysia, (tp, fp, fn, tn) India).
SCENARIOS = {
    "default": ((44, 34, 125, 604), (2, 2, 13, 29)),
    "screening": ((164, 349, 5, 289), (12, 19, 3, 12)),
    "fpr_dominant": ((7, 3, 162, 635), (0, 2, 15, 29)),
    "no_malaysian_positives": ((0, 0, 169, 638), (0, 2, 15, 29)),
}


def _reference_gaps(malaysia, india):
    """Newcombe intervals of each Malaysia-minus-India gap, from statsmodels."""
    def gap(k1, n1, k2, n2):
        low, high = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb", compare="diff")
        return k1 / n1 - k2 / n2, low, high

    (tp1, fp1, fn1, tn1), (tp2, fp2, fn2, tn2) = malaysia, india
    n1, n2 = tp1 + fp1 + fn1 + tn1, tp2 + fp2 + fn2 + tn2
    gaps = {
        "equal_opportunity_diff": gap(tp1, tp1 + fn1, tp2, tp2 + fn2),
        "fpr_diff": gap(fp1, fp1 + tn1, fp2, fp2 + tn2),
        "demographic_parity_diff": gap(tp1 + fp1, n1, tp2 + fp2, n2),
    }
    if tp1 + fp1 and tp2 + fp2:
        gaps["predictive_parity_diff"] = gap(tp1, tp1 + fp1, tp2, tp2 + fp2)
    tpr, fpr = gaps["equal_opportunity_diff"], gaps["fpr_diff"]
    gaps["equalized_odds_diff"] = tpr if abs(tpr[0]) >= abs(fpr[0]) else fpr
    return gaps


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_signed_gaps_and_newcombe_intervals(scenario):
    y, p, country = fold(*SCENARIOS[scenario])
    row = fairness_at_threshold(y, p, country, THRESHOLD)

    for name, (point, low, high) in _reference_gaps(*SCENARIOS[scenario]).items():
        reported = row[f"{name}_signed"] if name == "equalized_odds_diff" else row[name]
        assert reported == pytest.approx(point, abs=TOLERANCE), name
        assert row[f"{name}_ci_low"] == pytest.approx(low, abs=TOLERANCE), name
        assert row[f"{name}_ci_high"] == pytest.approx(high, abs=TOLERANCE), name


def _wilson_string(k, n):
    low, high = proportion_confint(k, n, method="wilson")
    return f"{k / n:.4f} ({low:.4f} - {high:.4f})"


def test_cohort_sizes_and_rate_strings():
    y, p, country = fold(*SCENARIOS["default"])
    row = fairness_at_threshold(y, p, country, THRESHOLD)

    assert row["n_malaysia"] == 807
    assert row["n_pos_malaysia"] == 169
    assert row["n_india"] == 46
    assert row["n_pos_india"] == 15
    assert row["tpr_malaysia_str"] == _wilson_string(44, 169)
    assert row["tpr_india_str"] == _wilson_string(2, 15)
    assert not row["equal_opportunity_diff_ci_excludes_0"]


def test_a_large_sensitivity_gap_is_distinguishable_from_zero():
    y, p, country = fold(*SCENARIOS["screening"])
    row = fairness_at_threshold(y, p, country, THRESHOLD)
    assert row["equal_opportunity_diff_ci_excludes_0"]
    assert row["tpr_malaysia_str"] == _wilson_string(164, 169)
    assert row["tpr_india_str"] == _wilson_string(12, 15)


def test_predictive_parity_is_flagged_when_india_barely_fires():
    """Fewer than ten Indian positive predictions -> flagged with '*'."""
    y, p, country = fold(*SCENARIOS["default"])
    row = fairness_at_threshold(y, p, country, THRESHOLD)
    assert row["n_positive_predictions_india"] == 4
    assert row["predictive_parity_diff_flag"] == "*"


def test_predictive_parity_is_not_estimable_with_no_positive_predictions():
    """No positive predictions in a cohort -> 'N/A'."""
    y, p, country = fold(*SCENARIOS["no_malaysian_positives"])
    row = fairness_at_threshold(y, p, country, THRESHOLD)
    assert row["n_positive_predictions_malaysia"] == 0
    assert row["predictive_parity_diff_flag"] == "N/A"
    assert np.isnan(row["predictive_parity_diff"])


def test_equalized_odds_switches_to_the_false_positive_gap_when_it_dominates():
    """When the FPR gap is larger than the TPR gap it carries the interval."""
    y, p, country = fold(*SCENARIOS["fpr_dominant"])
    row = fairness_at_threshold(y, p, country, THRESHOLD)
    assert row["equalized_odds_dominant_component"] == "fpr"
    # Reported as an absolute value; the interval stays signed and negative.
    assert row["equalized_odds_diff"] > 0
    assert row["equalized_odds_diff_signed"] < 0
    assert row["equalized_odds_diff_ci_high"] < 0


def test_equalized_odds_uses_the_sensitivity_gap_when_that_dominates():
    y, p, country = fold(*SCENARIOS["default"])
    row = fairness_at_threshold(y, p, country, THRESHOLD)
    assert row["equalized_odds_dominant_component"] == "tpr"


def test_points_fast_path_agrees_with_the_interval_bearing_path():
    """The bootstrap's fast path must not drift from the reported estimator."""
    y, p, country = fold(*SCENARIOS["default"])
    full = fairness_at_threshold(y, p, country, THRESHOLD)
    points = fairness_points(y, p, country, THRESHOLD)
    for name in ("equal_opportunity_diff", "demographic_parity_diff", "fpr_diff"):
        assert points[name] == pytest.approx(full[name])
    assert points["equalized_odds_diff"] == pytest.approx(full["equalized_odds_diff_signed"])


def test_the_sweep_covers_the_configured_grid():
    from sga.config import THRESHOLD_GRID

    y, p, country = fold(*SCENARIOS["default"])
    sweep = fairness_threshold_sweep(y, p, country)
    assert list(sweep["threshold"]) == [float(t) for t in THRESHOLD_GRID]
    assert sweep["threshold"].iloc[0] == 0.10
    assert sweep["threshold"].iloc[-1] == 0.90


def test_the_cohort_stratified_bootstrap_keeps_both_cohorts_in_every_replicate():
    """With 15 Indian events an unstratified resample loses the cohort entirely."""
    y, p, country = fold(*SCENARIOS["default"])
    bootstrap = bootstrap_fairness_sweep(
        y, p, country, thresholds=[THRESHOLD], n_boot=200
    )
    row = bootstrap.iloc[0]
    assert not np.isnan(row["equal_opportunity_diff_boot_ci_low"])
    assert (
        row["equal_opportunity_diff_boot_ci_low"]
        < row["equal_opportunity_diff_boot_ci_high"]
    )
    # The bootstrap interval is reported alongside the score interval, so it must
    # at least agree with it on the sign question at this cut-off.
    assert row["equal_opportunity_diff_boot_p"] > 0.05


def test_a_cohort_with_no_disparity_reports_zero_everywhere():
    counts = (40, 20, 60, 80)
    y, p, country = fold(counts, counts)
    row = fairness_at_threshold(y, p, country, THRESHOLD)
    for name in (
        "equal_opportunity_diff",
        "demographic_parity_diff",
        "predictive_parity_diff",
        "fpr_diff",
    ):
        assert row[name] == pytest.approx(0.0, abs=1e-12), name
    assert row["equalized_odds_diff"] == pytest.approx(0.0, abs=1e-12)
