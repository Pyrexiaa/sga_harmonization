"""Cohort-level fairness must reproduce the manuscript's Section 3.9 and Table S5.

The confusion counts below are reconstructed from the published rates and the
published stratum sizes (Malaysia 807 scans / 169 SGA, India 46 scans / 15 SGA),
and the expected differences and intervals are copied from the submitted
appendix. Together they pin down every choice the fairness code makes: signed
differences, Wilson rates, Newcombe gaps, which component equalized odds takes,
and when a difference is flagged as not reliably estimable.
"""

from __future__ import annotations

import numpy as np
import pytest

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


#: threshold -> ((tp, fp, fn, tn) Malaysia, (tp, fp, fn, tn) India), read off the
#: rates in appendix Table S4 against the published stratum sizes.
PUBLISHED_COUNTS = {
    0.50: ((43, 33, 126, 605), (2, 2, 13, 29)),
    0.10: ((164, 349, 5, 289), (12, 19, 3, 12)),
    0.75: ((7, 3, 162, 635), (0, 2, 15, 29)),
    0.85: ((0, 0, 169, 638), (0, 2, 15, 29)),
}

#: threshold -> {difference: (point, low, high)} from appendix Table S5.
#: Equalized odds is quoted as an ABSOLUTE value carrying the signed interval of
#: whichever error-rate gap attained the maximum.
PUBLISHED_DIFFERENCES = {
    0.50: {
        "equal_opportunity_diff": (0.1211, -0.1315, 0.2403),
        "equalized_odds_diff": (0.1211, -0.1315, 0.2403),
        "demographic_parity_diff": (0.0072, -0.1105, 0.0643),
        "predictive_parity_diff": (0.0658, -0.3016, 0.4313),
    },
    0.10: {
        "equal_opportunity_diff": (0.1704, 0.0355, 0.4228),
        "equalized_odds_diff": (0.1704, 0.0355, 0.4228),
        "demographic_parity_diff": (-0.0382, -0.1604, 0.1096),
        "predictive_parity_diff": (-0.0674, -0.2463, 0.0880),
    },
    0.75: {
        "equal_opportunity_diff": (0.0414, -0.1636, 0.0830),
        "equalized_odds_diff": (0.0598, -0.2025, -0.0123),
        "demographic_parity_diff": (-0.0311, -0.1331, 0.0020),
        "predictive_parity_diff": (0.7000, -0.0242, 0.8922),
    },
    0.85: {
        "equal_opportunity_diff": (0.0000, -0.2039, 0.0222),
        "equalized_odds_diff": (0.0645, -0.2072, -0.0175),
        "demographic_parity_diff": (-0.0435, -0.1453, -0.0117),
    },
}


@pytest.mark.parametrize("published_threshold", sorted(PUBLISHED_DIFFERENCES))
def test_table_s5_differences_and_intervals(published_threshold):
    y, p, country = fold(*PUBLISHED_COUNTS[published_threshold])
    row = fairness_at_threshold(y, p, country, THRESHOLD)

    for name, (point, low, high) in PUBLISHED_DIFFERENCES[published_threshold].items():
        reported = row[f"{name}_abs"] if name == "equalized_odds_diff" else row[name]
        assert reported == pytest.approx(point, abs=TOLERANCE), name
        assert row[f"{name}_ci_low"] == pytest.approx(low, abs=TOLERANCE), name
        assert row[f"{name}_ci_high"] == pytest.approx(high, abs=TOLERANCE), name


def test_section_3_9_narrative_numbers_at_the_default_cut_off():
    """"a true-positive rate of 0.2544 ... against 0.1333 ... " """
    y, p, country = fold(*PUBLISHED_COUNTS[0.50])
    row = fairness_at_threshold(y, p, country, THRESHOLD)

    assert row["n_malaysia"] == 807
    assert row["n_pos_malaysia"] == 169
    assert row["n_india"] == 46
    assert row["n_pos_india"] == 15
    assert row["tpr_malaysia_str"] == "0.2544 (0.1947 - 0.3251)"
    assert row["tpr_india_str"] == "0.1333 (0.0374 - 0.3788)"
    # The sensitivity gap's interval includes zero, which is exactly why the text
    # calls it a limitation rather than a precisely quantified effect.
    assert not row["equal_opportunity_diff_ci_excludes_0"]


def test_the_screening_threshold_makes_the_gap_distinguishable():
    """At 0.10 the Equal Opportunity Difference's interval excludes zero."""
    y, p, country = fold(*PUBLISHED_COUNTS[0.10])
    row = fairness_at_threshold(y, p, country, THRESHOLD)
    assert row["equal_opportunity_diff_ci_excludes_0"]
    assert row["tpr_malaysia_str"] == "0.9704 (0.9326 - 0.9873)"
    assert row["tpr_india_str"] == "0.8000 (0.5481 - 0.9295)"


def test_predictive_parity_is_flagged_when_india_barely_fires():
    """Fewer than ten Indian positive predictions -> the Table S5 '*' footnote."""
    y, p, country = fold(*PUBLISHED_COUNTS[0.50])
    row = fairness_at_threshold(y, p, country, THRESHOLD)
    assert row["n_positive_predictions_india"] == 4
    assert row["predictive_parity_diff_flag"] == "*"


def test_predictive_parity_is_not_estimable_with_no_positive_predictions():
    """No positive predictions in a cohort -> the Table S5 'N/A' footnote."""
    y, p, country = fold(*PUBLISHED_COUNTS[0.85])
    row = fairness_at_threshold(y, p, country, THRESHOLD)
    assert row["n_positive_predictions_malaysia"] == 0
    assert row["predictive_parity_diff_flag"] == "N/A"
    assert np.isnan(row["predictive_parity_diff"])


def test_equalized_odds_switches_to_the_false_positive_gap_when_it_dominates():
    """At 0.75 the FPR gap is larger than the TPR gap and carries the interval."""
    y, p, country = fold(*PUBLISHED_COUNTS[0.75])
    row = fairness_at_threshold(y, p, country, THRESHOLD)
    assert row["equalized_odds_dominant_component"] == "fpr"
    # Reported as an absolute value; the interval stays signed and negative.
    assert row["equalized_odds_diff"] > 0
    assert row["equalized_odds_diff_signed"] < 0
    assert row["equalized_odds_diff_ci_high"] < 0


def test_equalized_odds_uses_the_sensitivity_gap_when_that_dominates():
    y, p, country = fold(*PUBLISHED_COUNTS[0.50])
    row = fairness_at_threshold(y, p, country, THRESHOLD)
    assert row["equalized_odds_dominant_component"] == "tpr"


def test_points_fast_path_agrees_with_the_interval_bearing_path():
    """The bootstrap's fast path must not drift from the reported estimator."""
    y, p, country = fold(*PUBLISHED_COUNTS[0.50])
    full = fairness_at_threshold(y, p, country, THRESHOLD)
    points = fairness_points(y, p, country, THRESHOLD)
    for name in ("equal_opportunity_diff", "demographic_parity_diff", "fpr_diff"):
        assert points[name] == pytest.approx(full[name])
    assert points["equalized_odds_diff"] == pytest.approx(full["equalized_odds_diff_signed"])


def test_the_sweep_covers_exactly_the_published_grid():
    from sga.config import THRESHOLD_GRID

    y, p, country = fold(*PUBLISHED_COUNTS[0.50])
    sweep = fairness_threshold_sweep(y, p, country)
    assert list(sweep["threshold"]) == [float(t) for t in THRESHOLD_GRID]
    assert sweep["threshold"].iloc[0] == 0.10
    assert sweep["threshold"].iloc[-1] == 0.90


def test_the_cohort_stratified_bootstrap_keeps_both_cohorts_in_every_replicate():
    """With 15 Indian events an unstratified resample loses the cohort entirely."""
    y, p, country = fold(*PUBLISHED_COUNTS[0.50])
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
