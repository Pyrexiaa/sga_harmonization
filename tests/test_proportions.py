"""Score intervals must reproduce the intervals printed in the manuscript.

Every expected value below is copied from the submitted manuscript and appendix,
not from a previous run of this code, so these tests fail if the interval method
is ever changed to something that no longer produces the published numbers.

The external test fold, as reported in the Cohort-Level Fairness section:

    Malaysia   807 scans, 169 SGA  ->  638 AGA
    India       46 scans,  15 SGA  ->   31 AGA
"""

from __future__ import annotations

import pytest

from sga.evaluation.proportions import (
    newcombe_difference_ci,
    percentile_ci,
    rate_with_ci,
    wilson_ci,
)

TOLERANCE = 5e-5

#: (successes, n, point, low, high) - appendix Table S4.
TABLE_S4_WILSON = [
    # Sensitivity at the 0.50 cut-off.
    (43, 169, 0.2544, 0.1947, 0.3251),   # Malaysia
    (2, 15, 0.1333, 0.0374, 0.3788),     # India
    (45, 184, 0.2446, 0.1881, 0.3115),   # Total
    # Sensitivity at the 0.10 screening cut-off.
    (164, 169, 0.9704, 0.9326, 0.9873),  # Malaysia
    (12, 15, 0.8000, 0.5481, 0.9295),    # India
    # Specificity at the 0.50 cut-off.
    (605, 638, 0.9483, 0.9283, 0.9629),  # Malaysia
    (29, 31, 0.9355, 0.7928, 0.9821),    # India
    (634, 669, 0.9477, 0.9281, 0.9621),  # Total
    # Zero-count cells: the reason Wilson is used rather than a bootstrap.
    (0, 169, 0.0000, 0.0000, 0.0222),    # Malaysia sensitivity at 0.85
    (0, 15, 0.0000, 0.0000, 0.2039),     # India sensitivity at 0.65 and above
    # Saturated cell.
    (638, 638, 1.0000, 0.9940, 1.0000),  # Malaysia specificity at 0.80
]


@pytest.mark.parametrize("successes,n,point,low,high", TABLE_S4_WILSON)
def test_wilson_reproduces_appendix_table_s4(successes, n, point, low, high):
    interval = wilson_ci(successes, n)
    assert interval.point == pytest.approx(point, abs=TOLERANCE)
    assert interval.low == pytest.approx(low, abs=TOLERANCE)
    assert interval.high == pytest.approx(high, abs=TOLERANCE)


#: (k1, n1, k2, n2, point, low, high) - appendix Table S5, Malaysia minus India.
TABLE_S5_NEWCOMBE = [
    # Equal Opportunity Difference at 0.50: 0.1211 (-0.1315 to 0.2403).
    (43, 169, 2, 15, 0.1211, -0.1315, 0.2403),
    # Equal Opportunity Difference at 0.10: 0.1704 (0.0355 to 0.4228).
    (164, 169, 12, 15, 0.1704, 0.0355, 0.4228),
    # Demographic Parity Difference at 0.50: 0.0072 (-0.1105 to 0.0643).
    (76, 807, 4, 46, 0.0072, -0.1105, 0.0643),
    # Predictive Parity Difference at 0.10: -0.0674 (-0.2463 to 0.0880).
    (164, 513, 12, 31, -0.0674, -0.2463, 0.0880),
    # The FPR gap that becomes the Equalized Odds Difference at 0.75:
    # 0.0598 reported as an absolute value, with a wholly negative interval.
    (3, 638, 2, 31, -0.0598, -0.2025, -0.0123),
]


@pytest.mark.parametrize("k1,n1,k2,n2,point,low,high", TABLE_S5_NEWCOMBE)
def test_newcombe_reproduces_appendix_table_s5(k1, n1, k2, n2, point, low, high):
    interval = newcombe_difference_ci(k1, n1, k2, n2)
    assert interval.point == pytest.approx(point, abs=TOLERANCE)
    assert interval.low == pytest.approx(low, abs=TOLERANCE)
    assert interval.high == pytest.approx(high, abs=TOLERANCE)


def test_wilson_stays_inside_the_unit_interval_at_every_count():
    for n in (1, 5, 15, 46, 807):
        for successes in range(n + 1):
            interval = wilson_ci(successes, n)
            assert 0.0 <= interval.low <= interval.point <= interval.high <= 1.0


def test_newcombe_difference_is_signed_and_can_contain_zero():
    """The whole point of the signed interval: 'no gap' must be representable."""
    interval = newcombe_difference_ci(50, 100, 50, 100)
    assert interval.point == pytest.approx(0.0)
    assert interval.low < 0 < interval.high
    assert not interval.excludes(0.0)


def test_newcombe_detects_a_real_gap():
    interval = newcombe_difference_ci(90, 100, 10, 100)
    assert interval.excludes(0.0)


def test_an_empty_denominator_is_not_estimable_rather_than_zero():
    """PPV with no positive predictions is undefined, not 0.0."""
    interval = rate_with_ci(0, 0)
    assert not interval.is_estimable
    assert interval.format() == "N/A"
    assert not interval.excludes(0.0)


def test_interval_formatting_matches_the_table_layout():
    assert wilson_ci(43, 169).format() == "0.2544 (0.1947 - 0.3251)"


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_percentile_ci_ignores_nans_and_survives_an_empty_sample():
    import numpy as np

    low, high = percentile_ci([0.1, np.nan, 0.9, 0.5])
    assert low < high
    assert all(np.isnan(v) for v in percentile_ci([np.nan, np.nan]))
