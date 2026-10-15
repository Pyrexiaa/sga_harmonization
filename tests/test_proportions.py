"""Wilson and Newcombe score intervals, checked against statsmodels."""

from __future__ import annotations

import pytest
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

from sga.evaluation.proportions import (
    newcombe_difference_ci,
    percentile_ci,
    rate_with_ci,
    wilson_ci,
)

TOLERANCE = 1e-10

#: (successes, n), including zero-count and saturated cells.
WILSON_CASES = [
    (43, 169), (2, 15), (45, 184), (164, 169), (12, 15),
    (605, 638), (29, 31), (634, 669), (0, 169), (0, 15), (638, 638),
]


@pytest.mark.parametrize("successes,n", WILSON_CASES)
def test_wilson_matches_statsmodels(successes, n):
    interval = wilson_ci(successes, n)
    low, high = proportion_confint(successes, n, method="wilson")
    assert interval.point == pytest.approx(successes / n)
    assert interval.low == pytest.approx(low, abs=TOLERANCE)
    assert interval.high == pytest.approx(high, abs=TOLERANCE)


#: (k1, n1, k2, n2), first group minus second.
NEWCOMBE_CASES = [
    (43, 169, 2, 15),
    (164, 169, 12, 15),
    (76, 807, 4, 46),
    (164, 513, 12, 31),
    (3, 638, 2, 31),
    (0, 169, 0, 15),
    (638, 638, 31, 31),
]


@pytest.mark.parametrize("k1,n1,k2,n2", NEWCOMBE_CASES)
def test_newcombe_matches_statsmodels_hybrid_score(k1, n1, k2, n2):
    interval = newcombe_difference_ci(k1, n1, k2, n2)
    low, high = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb", compare="diff")
    assert interval.point == pytest.approx(k1 / n1 - k2 / n2)
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


def test_interval_formatting():
    assert wilson_ci(43, 169).format() == "0.2544 (0.1947 - 0.3251)"


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_percentile_ci_ignores_nans_and_survives_an_empty_sample():
    import numpy as np

    low, high = percentile_ci([0.1, np.nan, 0.9, 0.5])
    assert low < high
    assert all(np.isnan(v) for v in percentile_ci([np.nan, np.nan]))
