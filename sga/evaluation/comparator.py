"""The EFW < 10th centile rule as a clinical comparator (appendix Table S9, Figure 4).

The rule is a fixed binary decision, so it yields one confusion matrix. The model is
read at its own cut-offs and also at the cut-off that flags the same proportion of
scans as the rule, so the two can be compared at equal clinical workload.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sga.evaluation.metrics import confusion_counts, rate_numerator_denominator
from sga.evaluation.proportions import rate_with_ci

OPERATING_POINT_RATES = ("sensitivity", "specificity", "ppv", "npv")


def centile_risk(centile):
    """Rank-equivalent risk score for a centile: smaller centile, higher risk."""
    return 1.0 - np.asarray(centile, dtype=float) / 100.0


def operating_point(y_true, flagged):
    """Confusion counts, Wilson-bounded rates and the flagged rate of one decision."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(flagged).astype(int)
    counts = confusion_counts(y_true, flagged)
    row = dict(counts)
    for rate in OPERATING_POINT_RATES:
        interval = rate_with_ci(*rate_numerator_denominator(counts, rate))
        row[rate] = interval.point
        row[f"{rate}_ci_low"] = interval.low
        row[f"{rate}_ci_high"] = interval.high
    row["flagged_rate"] = float(flagged.mean()) if len(flagged) else float("nan")
    return row


def workload_matched_threshold(y_prob, flagged_rate):
    """Model cut-off that flags ``flagged_rate`` of the scans (Table S9b)."""
    if not 0 < flagged_rate < 1:
        return float("nan")
    return float(np.quantile(np.asarray(y_prob, dtype=float), 1 - flagged_rate))


def net_benefit(y_true, flagged, threshold):
    """``TP/n - FP/n * pt/(1 - pt)`` of a fixed set of flagged scans."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(flagged).astype(int)
    n = len(y_true)
    if n == 0 or threshold >= 1:
        return float("nan")
    counts = confusion_counts(y_true, flagged)
    return float(counts["tp"] / n - counts["fp"] / n * threshold / (1 - threshold))


def rule_net_benefit(y_true, flagged, thresholds):
    """Net benefit of a fixed rule across ``thresholds`` (its Figure 4 curve)."""
    return [net_benefit(y_true, flagged, t) for t in thresholds]


def operating_point_table(y_true, y_prob, flagged_by_rule, thresholds, rule_name):
    """Table S9b for one cohort: the rule, the model at each cut-off, and at matched workload."""
    rows = [{"strategy": rule_name, "threshold": float("nan"), **operating_point(y_true, flagged_by_rule)}]
    for threshold in thresholds:
        rows.append(
            {
                "strategy": f"model at {threshold:g}",
                "threshold": float(threshold),
                **operating_point(y_true, np.asarray(y_prob) >= threshold),
            }
        )
    matched = workload_matched_threshold(y_prob, float(np.mean(flagged_by_rule)))
    if np.isfinite(matched):
        rows.append(
            {
                "strategy": "model at matched workload",
                "threshold": matched,
                **operating_point(y_true, np.asarray(y_prob) >= matched),
            }
        )
    return pd.DataFrame(rows)
