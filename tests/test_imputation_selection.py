"""Retention rules for the cross-domain imputed features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from sga import config
from sga.imputation.apply import (
    _multiclass_gate,
    passes_retention_rule,
    select_features_with_threshold,
)
from sga.imputation.diagnostics import (
    class_distribution,
    majority_baseline_f1,
    majority_class,
    multiclass_quality,
)
from sga.imputation.train_imputers import MULTICLASS_GATE_FILE, write_multiclass_gate


@pytest.mark.parametrize(
    "kind, primary, secondary, expected",
    [
        ("binary", 0.81, 0.51, True),
        ("binary", 0.80, 0.90, False),
        ("binary", 0.88, 0.12, False),
        ("regression", 0.91, None, True),
        ("regression", 0.80, None, False),
    ],
)
def test_retention_bounds_are_strict(kind, primary, secondary, expected):
    assert passes_retention_rule(kind, primary, secondary) is expected


def test_multiclass_rule_needs_the_baseline_and_the_auroc_interval():
    below_baseline = {"baseline_f1": 0.9085, "auroc_ci": (0.448, 0.623)}
    assert not passes_retention_rule("multiclass", 0.8875, 0.534, below_baseline)
    informative = {"baseline_f1": 0.85, "auroc_ci": (0.55, 0.70)}
    assert passes_retention_rule("multiclass", 0.90, 0.62, informative)
    assert not passes_retention_rule("multiclass", 0.90, 0.62, None)


def test_constant_features_are_excluded_not_silently_retained():
    selected, removed = select_features_with_threshold(
        list(config.CONSTANT_ZERO_FEATURES), selection_type="binary"
    )
    assert selected == []
    assert sorted(removed) == sorted(config.CONSTANT_ZERO_FEATURES)


def test_majority_baseline_equals_weighted_f1_of_constant_rule():
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    assert majority_class(y) == 0
    # Weighted F1 of "always 0" = 0.80 * F1(class 0) = 0.80 * 2*0.8/(1+0.8).
    assert majority_baseline_f1(y, 0) == pytest.approx(0.8 * 2 * 0.8 / 1.8)


def test_imputer_that_only_predicts_the_majority_fails_the_gate():
    rng = np.random.RandomState(0)
    y = rng.choice([0, 1, 2], size=600, p=[0.85, 0.10, 0.05])
    proba = rng.dirichlet([1, 1, 1], size=len(y))
    quality = multiclass_quality(y, np.zeros_like(y), proba, [0, 1, 2], n_boot=200)

    assert quality.weighted_f1 == pytest.approx(quality.baseline_f1)
    assert not quality.beats_majority_baseline
    assert quality.macro_auroc_low < 0.5 < quality.macro_auroc_high
    assert not quality.retained
    assert quality.recall[0] == 1.0 and quality.recall[1] == 0.0 and quality.recall[2] == 0.0
    assert quality.confusion_frame().loc["observed 0", "predicted 0"] == (y == 0).sum()


def test_informative_imputer_passes_the_gate():
    rng = np.random.RandomState(1)
    y = rng.choice([0, 1, 2], size=900, p=[0.5, 0.3, 0.2])
    logits = rng.normal(size=(len(y), 3))
    logits[np.arange(len(y)), y] += 3.0
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    quality = multiclass_quality(y, proba.argmax(axis=1), proba, [0, 1, 2], n_boot=200)

    assert quality.beats_majority_baseline
    assert quality.auroc_ci_excludes_chance
    assert quality.retained
    summary = quality.summary_rows()
    assert summary["passes"].tolist() == [True, True]


def test_majority_is_taken_from_the_training_labels_when_given():
    y_val = np.array([1, 1, 1, 0])
    quality = multiclass_quality(
        y_val, y_val, np.eye(2)[y_val], [0, 1], train_y=np.array([0, 0, 0, 1]), n_boot=10
    )
    assert quality.majority == 0
    assert quality.baseline_f1 == pytest.approx(majority_baseline_f1(y_val, 0))


def test_class_distribution_counts_each_level_per_partition():
    frames = {
        "Malaysia": pd.DataFrame({"af": [0, 0, 1, np.nan]}),
        "India (imputed)": pd.DataFrame({"af": [0, 2]}),
    }
    table = class_distribution(frames, "af").set_index("partition")
    assert table.loc["Malaysia", "total"] == 3
    assert table.loc["Malaysia", "n_2"] == 0
    assert table.loc["India (imputed)", "pct_2"] == pytest.approx(50.0)


def test_multiclass_gate_round_trips_through_the_retention_rule(tmp_path):
    rng = np.random.RandomState(2)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = pd.Series((X["a"] > 0).astype(int) + (X["b"] > 1).astype(int), name="af")
    validation = X.assign(af=y)
    model = LogisticRegression(max_iter=500).fit(X, y)

    quality = write_multiclass_gate(model, validation, ["a", "b"], "af", y, str(tmp_path))
    assert (tmp_path / "metrics" / MULTICLASS_GATE_FILE).exists()

    gate = _multiclass_gate(str(tmp_path))
    assert gate["baseline_f1"] == pytest.approx(quality.baseline_f1)
    assert gate["auroc_ci"] == pytest.approx((quality.macro_auroc_low, quality.macro_auroc_high))
    assert passes_retention_rule("multiclass", quality.weighted_f1, quality.macro_auroc, gate) == quality.retained


def test_multiclass_feature_without_a_gate_file_is_not_retained(tmp_path):
    assert _multiclass_gate(str(tmp_path)) is None
    assert not passes_retention_rule("multiclass", 0.99, 0.99, gate=None)


def test_multiclass_rule_rejects_auroc_interval_spanning_chance():
    gate = {"baseline_f1": 0.5, "auroc_ci": (0.45, 0.80)}
    assert not passes_retention_rule("multiclass", 0.9, 0.9, gate)
    gate = {"baseline_f1": 0.5, "auroc_ci": (0.55, 0.80)}
    assert passes_retention_rule("multiclass", 0.9, 0.9, gate)
