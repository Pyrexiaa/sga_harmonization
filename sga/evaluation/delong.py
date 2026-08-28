"""DeLong's test for two correlated ROC curves, with Holm-Bonferroni correction.

The manuscript reports the pairwise comparisons "after Holm-Bonferroni
correction" (Table 6), so that is the name used throughout here. Holm-Bonferroni
is the step-down procedure: order the family's p-values ascending, multiply the
k-th by (m - k + 1), then enforce monotonicity and cap at 1. It controls the
family-wise error rate exactly as the single-step Bonferroni correction does but
is uniformly more powerful, and it is verified against
``statsmodels.stats.multitest.multipletests(method="holm")`` in
``tests/test_evaluation.py``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def compute_midrank(x):
    """Midranks of ``x``, ties receiving the average of their ranks."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def delong_test(ground_truth, pred1, pred2):
    """Paired DeLong test of two AUROCs computed on the SAME rows."""
    y = np.asarray(ground_truth, dtype=np.int32)
    p1 = np.asarray(pred1, dtype=np.float64)
    p2 = np.asarray(pred2, dtype=np.float64)
    positive, negative = y == 1, y == 0
    m, n = int(positive.sum()), int(negative.sum())
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan, np.nan

    def placement(scores):
        pos, neg = scores[positive], scores[negative]
        v10 = np.array([np.mean(pos[i] > neg) + 0.5 * np.mean(pos[i] == neg) for i in range(m)])
        v01 = np.array([np.mean(neg[j] < pos) + 0.5 * np.mean(neg[j] == pos) for j in range(n)])
        return v10.mean(), v10, v01

    auc1, v10_1, v01_1 = placement(p1)
    auc2, v10_2, v01_2 = placement(p2)

    s10 = np.cov(np.column_stack([v10_1, v10_2]).T) if m > 1 else np.zeros((2, 2))
    s01 = np.cov(np.column_stack([v01_1, v01_2]).T) if n > 1 else np.zeros((2, 2))
    covariance = s10 / m + s01 / n
    variance = covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]

    if variance <= 0:
        if abs(auc1 - auc2) < 1e-10:
            return auc1, auc2, 0.0, 1.0
        return auc1, auc2, np.nan, np.nan

    z = (auc1 - auc2) / np.sqrt(variance)
    return auc1, auc2, z, 2.0 * (1.0 - stats.norm.cdf(abs(z)))


def holm_bonferroni_correction(p_values):
    """Holm-Bonferroni step-down adjusted p-values, preserving the input order.

    NaNs are passed through untouched and excluded from the family size, so a
    comparison that could not be computed does not inflate the correction applied
    to the ones that could.
    """
    p_values = np.asarray(p_values, dtype=float)
    finite = ~np.isnan(p_values)
    adjusted = np.full_like(p_values, np.nan)

    values = p_values[finite]
    order = np.argsort(values)
    m = len(values)
    running = 0.0
    corrected = np.empty(m)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * values[idx])
        corrected[idx] = min(1.0, running)

    adjusted[finite] = corrected
    return adjusted


#: Deprecated alias. The procedure is Holm-Bonferroni; the manuscript names it in
#: full, so new code should call ``holm_bonferroni_correction``.
holm_correction = holm_bonferroni_correction


def pairwise_delong(y_true, probabilities, correct=True):
    """All pairwise DeLong comparisons between named probability vectors.

    With ``correct=True`` the family is every pair in this call, and the adjusted
    values land in ``p_value_holm_bonferroni`` (manuscript Table 6). The
    uncorrected values remain in ``p_value``.
    """
    rows = []
    for name_a, name_b in itertools.combinations(probabilities, 2):
        auc_a, auc_b, z, p = delong_test(y_true, probabilities[name_a], probabilities[name_b])
        rows.append(
            {
                "model_1": name_a,
                "model_2": name_b,
                "auc_1": auc_a,
                "auc_2": auc_b,
                "auc_diff": auc_a - auc_b,
                "z": z,
                "p_value": p,
            }
        )

    result = pd.DataFrame(rows)
    if correct and not result.empty:
        result["p_value_holm_bonferroni"] = holm_bonferroni_correction(
            result["p_value"].to_numpy()
        )
    return result
