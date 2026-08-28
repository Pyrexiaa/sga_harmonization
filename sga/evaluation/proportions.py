"""Score-based confidence intervals for proportions and their differences.

The manuscript's Statistical Analysis section specifies these, not the percentile
bootstrap, for every quantity that is a proportion:

    "Confidence intervals for proportions, including sensitivity, specificity,
     positive and negative predictive value and cohort-level true and false
     positive rates, were computed by the Wilson score method. The intervals for
     differences between two independent proportions were computed by Newcombe's
     hybrid score method. Both remain bounded and well-calibrated at the small
     event counts present in the Indian stratum, where a percentile bootstrap can
     return degenerate intervals."

The bootstrap (``sga.evaluation.bootstrap``) remains the method for AUROC, AUPRC,
ECE and Brier, which are not proportions.

Why this matters numerically. With 15 Indian SGA events, a percentile bootstrap of
an ABSOLUTE difference is biased away from zero and can return an interval that
cannot cover 0 even when the two rates are indistinguishable. Wilson and Newcombe
are closed-form, bounded to [0, 1] and [-1, 1] respectively, and stay sensible at
zero-count cells, where the bootstrap degenerates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import norm

from sga.config import ALPHA


@dataclass(frozen=True)
class Interval:
    """A point estimate with its lower and upper confidence bound."""

    point: float
    low: float
    high: float

    @property
    def is_estimable(self) -> bool:
        """False when the estimate or either bound is undefined."""
        return not any(
            v is None or (isinstance(v, float) and math.isnan(v))
            for v in (self.point, self.low, self.high)
        )

    def format(self, decimals: int = 4, na: str = "N/A") -> str:
        """``0.2544 (0.1947 - 0.3251)`` - the layout used in Tables S4 and S5."""
        if not self.is_estimable:
            return na
        return (
            f"{self.point:.{decimals}f} "
            f"({self.low:.{decimals}f} - {self.high:.{decimals}f})"
        )

    def as_dict(self, prefix: str) -> dict:
        """Flatten into ``{prefix, prefix_ci_low, prefix_ci_high, prefix_str}``."""
        return {
            prefix: self.point,
            f"{prefix}_ci_low": self.low,
            f"{prefix}_ci_high": self.high,
            f"{prefix}_str": self.format(),
        }

    def excludes(self, value: float = 0.0) -> bool:
        """True when the interval lies wholly above or wholly below ``value``."""
        if not self.is_estimable:
            return False
        return bool(self.low > value or self.high < value)


NOT_ESTIMABLE = Interval(float("nan"), float("nan"), float("nan"))


@lru_cache(maxsize=8)
def _z(alpha: float) -> float:
    """Two-sided normal quantile for a ``1 - alpha`` interval (cached: the
    bootstrap comparison calls this tens of thousands of times)."""
    return float(norm.ppf(1 - alpha / 2))


def wilson_ci(successes, n, alpha: float = ALPHA) -> Interval:
    """Wilson score interval for a binomial proportion.

    ``successes`` events out of ``n`` trials. Unlike the Wald interval this never
    leaves [0, 1] and stays defined at ``successes`` of 0 or ``n`` - the cases that
    actually occur in the Indian stratum (for example 0/15 at the 0.85 cut-off,
    which Table S4 reports as 0.0000 (0.0000 - 0.2039)).
    """
    n = int(n)
    if n <= 0:
        return NOT_ESTIMABLE

    successes = int(round(float(successes)))
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} outside [0, {n}]")

    z = _z(alpha)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Interval(p, max(0.0, centre - half_width), min(1.0, centre + half_width))


def newcombe_difference_ci(
    successes_a, n_a, successes_b, n_b, alpha: float = ALPHA
) -> Interval:
    """Newcombe's hybrid score interval for ``p_a - p_b`` (independent samples).

    Newcombe (1998) method 10: take the Wilson interval for each proportion
    separately and combine the bounds that move the difference furthest,

        lower = (p_a - p_b) - sqrt((p_a - l_a)^2 + (u_b - p_b)^2)
        upper = (p_a - p_b) + sqrt((u_a - p_a)^2 + (p_b - l_b)^2)

    The interval is SIGNED and therefore free to contain 0, which is what makes
    "is there a gap at all" a real question rather than one the interval has
    already answered by construction.
    """
    a = wilson_ci(successes_a, n_a, alpha)
    b = wilson_ci(successes_b, n_b, alpha)
    if not (a.is_estimable and b.is_estimable):
        return NOT_ESTIMABLE

    difference = a.point - b.point
    low = difference - math.sqrt((a.point - a.low) ** 2 + (b.high - b.point) ** 2)
    high = difference + math.sqrt((a.high - a.point) ** 2 + (b.point - b.low) ** 2)
    return Interval(difference, max(-1.0, low), min(1.0, high))


def rate_with_ci(numerator, denominator, alpha: float = ALPHA) -> Interval:
    """Wilson interval for ``numerator / denominator``, or NOT_ESTIMABLE at 0/0.

    Rates such as PPV are undefined when nothing is predicted positive. Returning
    an explicit non-estimable interval keeps that distinct from a genuine 0.0,
    which is the distinction appendix Table S5 marks as "N/A".
    """
    denominator = int(denominator)
    if denominator <= 0:
        return NOT_ESTIMABLE
    return wilson_ci(numerator, denominator, alpha)


def difference_with_ci(
    numerator_a, denominator_a, numerator_b, denominator_b, alpha: float = ALPHA
) -> Interval:
    """Newcombe interval for ``a/n_a - b/n_b``, or NOT_ESTIMABLE at an empty cell."""
    if int(denominator_a) <= 0 or int(denominator_b) <= 0:
        return NOT_ESTIMABLE
    return newcombe_difference_ci(
        numerator_a, denominator_a, numerator_b, denominator_b, alpha
    )


def format_interval(interval: Interval, decimals: int = 4, na: str = "N/A") -> str:
    """Module-level alias of :meth:`Interval.format` for callers holding a value."""
    return interval.format(decimals=decimals, na=na)


def percentile_ci(values, alpha: float = ALPHA):
    """Percentile bounds of ``values``, ignoring NaNs. ``(nan, nan)`` when empty.

    Used only for the bootstrap intervals reported ALONGSIDE the score intervals,
    never for the primary numbers.
    """
    finite = np.asarray([v for v in np.asarray(values, dtype=float) if not np.isnan(v)])
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


__all__ = [
    "Interval",
    "NOT_ESTIMABLE",
    "difference_with_ci",
    "format_interval",
    "newcombe_difference_ci",
    "percentile_ci",
    "rate_with_ci",
    "wilson_ci",
]
