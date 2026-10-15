"""Quality diagnostics for a multiclass cross-domain imputer (appendix Table S2c).

A weighted F1 is governed by the majority level once one level dominates the target,
so an imputer that never predicts a minority level can still score highly; and a
macro one-vs-rest AUROC over unordered levels of very unequal prevalence is too noisy
to read without its confidence interval. The multiclass retention rule therefore adds
two checks to the F1 and AUROC thresholds: the weighted F1 must beat an
always-predict-majority baseline, and the macro one-vs-rest AUROC 95% CI must exclude
0.50.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from sga.config import ALPHA, MULTICLASS_AUROC_THRESHOLD, MULTICLASS_F1_THRESHOLD, N_BOOTSTRAP, SEED


def majority_class(y):
    """Most frequent level of ``y`` (ties broken by the smallest level)."""
    values, counts = np.unique(np.asarray(y).astype(int), return_counts=True)
    return int(values[np.argmax(counts)])


def majority_baseline_f1(y_true, majority):
    """Weighted F1 of a rule that always predicts ``majority``."""
    y_true = np.asarray(y_true).astype(int)
    return float(
        f1_score(y_true, np.full_like(y_true, majority), average="weighted", zero_division=0)
    )


def _macro_ovr_auroc(y_true, proba, classes):
    """Macro one-vs-rest AUROC, undefined unless every level is present."""
    if not np.isin(classes, y_true).all():
        return float("nan")
    try:
        return float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
        )
    except ValueError:
        return float("nan")


def macro_ovr_auroc_ci(y_true, proba, classes, n_boot=N_BOOTSTRAP, alpha=ALPHA, seed=SEED):
    """Macro one-vs-rest AUROC with a percentile bootstrap interval."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    classes = np.asarray(classes).astype(int)
    point = _macro_ovr_auroc(y_true, proba, classes)

    rng = np.random.RandomState(seed)
    replicates = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        value = _macro_ovr_auroc(y_true[idx], proba[idx], classes)
        if not np.isnan(value):
            replicates.append(value)
    if not replicates:
        return point, float("nan"), float("nan")
    low, high = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(low), float(high)


@dataclass
class MulticlassQuality:
    """Everything the multiclass retention rule and Table S2c report."""

    weighted_f1: float
    baseline_f1: float
    macro_auroc: float
    macro_auroc_low: float
    macro_auroc_high: float
    majority: int
    labels: list = field(default_factory=list)
    confusion: np.ndarray | None = None
    recall: dict = field(default_factory=dict)

    @property
    def beats_majority_baseline(self) -> bool:
        return bool(self.weighted_f1 > self.baseline_f1)

    @property
    def auroc_ci_excludes_chance(self) -> bool:
        return bool(
            np.isfinite(self.macro_auroc_low) and np.isfinite(self.macro_auroc_high)
            and (self.macro_auroc_low > 0.5 or self.macro_auroc_high < 0.5)
        )

    @property
    def retained(self) -> bool:
        return bool(
            self.weighted_f1 > MULTICLASS_F1_THRESHOLD
            and self.macro_auroc > MULTICLASS_AUROC_THRESHOLD
            and self.beats_majority_baseline
            and self.auroc_ci_excludes_chance
        )

    def summary_rows(self):
        """Table S2c(c): the imputer against the always-predict-majority baseline."""
        return pd.DataFrame(
            [
                {
                    "metric": "Weighted F1",
                    "imputer": self.weighted_f1,
                    "baseline": self.baseline_f1,
                    "passes": self.beats_majority_baseline,
                },
                {
                    "metric": "Macro one-vs-rest AUROC",
                    "imputer": self.macro_auroc,
                    "imputer_ci_low": self.macro_auroc_low,
                    "imputer_ci_high": self.macro_auroc_high,
                    "baseline": 0.5,
                    "passes": self.auroc_ci_excludes_chance,
                },
            ]
        )

    def confusion_frame(self):
        """Table S2c(b): rows observed, columns predicted, with per-class recall."""
        frame = pd.DataFrame(
            self.confusion,
            index=[f"observed {c}" for c in self.labels],
            columns=[f"predicted {c}" for c in self.labels],
        )
        frame["recall"] = [self.recall[c] for c in self.labels]
        return frame


def multiclass_quality(y_true, y_pred, proba, classes, train_y=None, n_boot=N_BOOTSTRAP, seed=SEED):
    """Score a multiclass imputer on held-out rows.

    The majority level is taken from ``train_y`` when given, so the baseline is the
    rule a practitioner would actually have available.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    classes = np.asarray(classes).astype(int)
    majority = majority_class(train_y if train_y is not None else y_true)

    labels = sorted(set(classes.tolist()) | set(np.unique(y_true).tolist()))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    support = matrix.sum(axis=1)
    recall = {
        label: float(matrix[i, i] / support[i]) if support[i] else float("nan")
        for i, label in enumerate(labels)
    }
    auroc, low, high = macro_ovr_auroc_ci(y_true, proba, classes, n_boot=n_boot, seed=seed)
    return MulticlassQuality(
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        baseline_f1=majority_baseline_f1(y_true, majority),
        macro_auroc=auroc,
        macro_auroc_low=low,
        macro_auroc_high=high,
        majority=majority,
        labels=labels,
        confusion=matrix,
        recall=recall,
    )


def class_distribution(frames, target):
    """Table S2c(a)/(d): ``n (%)`` of each level of ``target`` per named frame."""
    levels = sorted(
        {int(v) for frame in frames.values() for v in pd.Series(frame[target]).dropna().unique()}
    )
    rows = []
    for name, frame in frames.items():
        values = pd.Series(frame[target]).dropna().astype(int)
        total = len(values)
        row = {"partition": name, "total": total}
        for level in levels:
            n = int((values == level).sum())
            row[f"n_{level}"] = n
            row[f"pct_{level}"] = 100.0 * n / total if total else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
