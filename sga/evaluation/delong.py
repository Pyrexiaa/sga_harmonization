"""DeLong's tests for two ROC curves, with Holm-Bonferroni correction.

DeLong's test treats scans as independent; it does not account for repeated scans
from the same pregnancy, which the pregnancy-cluster bootstrap does.

Pairwise comparisons (Table 6) are corrected by Holm-Bonferroni, the step-down
procedure: order the family's p-values ascending, multiply the
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


def _delong_auc_variance(ground_truth, scores):
    """AUROC and its DeLong variance for one ROC curve."""
    y = np.asarray(ground_truth, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.nan, np.nan
    v10 = np.array([np.mean(x > neg) + 0.5 * np.mean(x == neg) for x in pos])
    v01 = np.array([np.mean(x < pos) + 0.5 * np.mean(x == pos) for x in neg])
    variance = (np.var(v10, ddof=1) / m if m > 1 else 0.0) + (
        np.var(v01, ddof=1) / n if n > 1 else 0.0
    )
    return float(v10.mean()), float(variance)


def delong_test_unpaired(ground_truth_1, pred_1, ground_truth_2, pred_2):
    """DeLong test of two AUROCs computed on DISJOINT sets of records.

    With independent samples the covariance term of the paired test is absent and
    the two variances add (appendix Table S10c).
    """
    auc1, var1 = _delong_auc_variance(ground_truth_1, pred_1)
    auc2, var2 = _delong_auc_variance(ground_truth_2, pred_2)
    if any(np.isnan(v) for v in (auc1, var1, auc2, var2)) or var1 + var2 <= 0:
        return auc1, auc2, np.nan, np.nan
    z = (auc1 - auc2) / np.sqrt(var1 + var2)
    return auc1, auc2, float(z), float(2.0 * (1.0 - stats.norm.cdf(abs(z))))


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


def pairwise_delong(y_true, probabilities, correct=True):
    """All pairwise DeLong comparisons between named probability vectors.

    With ``correct=True`` the family is every pair in this call, and the adjusted
    values land in ``p_value_holm_bonferroni`` (Table 6). The
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
