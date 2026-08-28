"""Discrimination, calibration and threshold-dependent metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sga.config import DECISION_THRESHOLD, ECE_BINS

EVAL_SPLITS = ["total", "malaysia", "india"]

METRIC_COLUMNS = [
    "balanced_accuracy",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "auprc",
    "brier_score",
    "ece",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
]


def expected_calibration_error(y_true, y_prob, n_bins=ECE_BINS):
    """ECE over ``n_bins`` equal-width probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")

    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        upper_inclusive = i == n_bins - 1
        in_bin = (y_prob >= edges[i]) & (
            y_prob <= edges[i + 1] if upper_inclusive else y_prob < edges[i + 1]
        )
        if in_bin.sum() == 0:
            continue
        ece += in_bin.sum() / len(y_true) * abs(y_true[in_bin].mean() - y_prob[in_bin].mean())
    return ece


def confusion_counts(y_true, y_pred):
    """Raw ``tp/fp/fn/tn`` counts from a binary confusion matrix.

    The counts, not just the rates, are what the Wilson and Newcombe intervals in
    ``sga.evaluation.proportions`` are computed from, so every reporting path that
    needs a score interval reads them from here.

    Counted with plain NumPy rather than ``sklearn.metrics.confusion_matrix``: the
    fairness bootstrap calls this tens of thousands of times, and the validation and
    label-alignment work sklearn does on every call dominates the runtime there.
    An empty input yields four zeros, matching the previous behaviour.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    both = y_true & y_pred
    return {
        "tp": int(both.sum()),
        "fp": int((~y_true & y_pred).sum()),
        "fn": int((y_true & ~y_pred).sum()),
        "tn": int((~y_true & ~y_pred).sum()),
    }


#: numerator and denominator of each threshold-dependent rate, in counts.
RATE_COUNTS = {
    "sensitivity": ("tp", ("tp", "fn")),
    "specificity": ("tn", ("tn", "fp")),
    "ppv": ("tp", ("tp", "fp")),
    "npv": ("tn", ("tn", "fn")),
}


def rate_numerator_denominator(counts, rate):
    """``(numerator, denominator)`` of one rate, given a confusion-count dict."""
    try:
        numerator_key, denominator_keys = RATE_COUNTS[rate]
    except KeyError as exc:
        raise ValueError(
            f"Unknown rate {rate!r}; choose from {sorted(RATE_COUNTS)}"
        ) from exc
    return counts[numerator_key], sum(counts[k] for k in denominator_keys)


def confusion_rates(y_true, y_pred, undefined=0.0):
    """Sensitivity, specificity, PPV and NPV from a binary confusion matrix.

    ``undefined`` is returned for a rate whose denominator is empty. It defaults to
    ``0.0`` for backwards compatibility with the aggregate metric tables; pass
    ``float("nan")`` where an empty denominator must stay distinguishable from a
    genuine zero (cohort-level reporting does exactly that).
    """
    counts = confusion_counts(y_true, y_pred)
    rates = {}
    for rate in RATE_COUNTS:
        numerator, denominator = rate_numerator_denominator(counts, rate)
        rates[rate] = numerator / denominator if denominator else undefined
    return rates


def basic_metrics(y_true, y_prob, threshold=DECISION_THRESHOLD):
    """Compact metric set used by the analysis scripts."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    both_classes = len(np.unique(y_true)) > 1
    out = {
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "auroc": roc_auc_score(y_true, y_prob) if both_classes else float("nan"),
        "auprc": average_precision_score(y_true, y_prob) if both_classes else float("nan"),
    }
    out.update(confusion_counts(y_true, y_pred))
    out.update(confusion_rates(y_true, y_pred))
    try:
        out["brier"] = brier_score_loss(y_true, y_prob)
    except ValueError:
        out["brier"] = float("nan")
    out["ece"] = expected_calibration_error(y_true, y_prob)
    return out


def full_metrics(y_true, y_pred, y_prob):
    """Complete metric row used for the per-country result tables."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        return {k: float("nan") for k in METRIC_COLUMNS} | {"n_samples": 0}

    both_classes = len(np.unique(y_true)) > 1
    metrics = {
        "n_samples": int(len(y_true)),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob) if both_classes else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auprc": average_precision_score(y_true, y_prob) if both_classes else float("nan"),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    metrics.update(confusion_rates(y_true, y_pred))
    return metrics


def youden_threshold(y_true, y_prob):
    """Operating point maximising sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def threshold_for_sensitivity(y_true, y_prob, target=0.80):
    """Highest probability cut-off that still reaches ``target`` sensitivity."""
    from sklearn.metrics import roc_curve

    _, tpr, thresholds = roc_curve(y_true, y_prob)
    reaching = np.where(tpr >= target)[0]
    return float(thresholds[reaching[0]]) if len(reaching) else 0.0
