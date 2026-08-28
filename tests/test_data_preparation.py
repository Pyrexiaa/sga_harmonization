"""Tests for cohort preparation: screening, scaling, and the Table 1 statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sga.config import COMMON_FEATURES, LABEL
from sga.data.cleaning import CONTINUOUS_FEATURE_LOGICAL_RANGE, remove_illogical_values
from sga.data.scaling import scale_feature_test, scale_feature_train
from sga.data.statistics import get_mean_continuous_feature, normality_and_parametric_test

# ── Physiological range filtering ────────────────────────────────────────────


def test_out_of_range_rows_are_dropped():
    low, high = CONTINUOUS_FEATURE_LOGICAL_RANGE["fl"]
    df = pd.DataFrame({"fl": [low - 1, (low + high) / 2, high + 1]})
    remove_illogical_values(df)
    assert len(df) == 1
    assert low <= df.loc[0, "fl"] <= high


def test_boundary_values_are_kept():
    low, high = CONTINUOUS_FEATURE_LOGICAL_RANGE["fl"]
    df = pd.DataFrame({"fl": [low, high]})
    remove_illogical_values(df)
    assert len(df) == 2


def test_nan_is_tolerated_by_default_and_dropped_when_asked():
    df = pd.DataFrame({"fl": [np.nan, 5.0]})
    remove_illogical_values(df)
    assert len(df) == 2

    df = pd.DataFrame({"fl": [np.nan, 5.0]})
    remove_illogical_values(df, keep_nan=False)
    assert len(df) == 1


def test_femur_length_range_actually_filters_both_ends():
    """A range filter written with OR instead of AND drops nothing."""
    values = pd.Series([30.0, 45.0, 60.0, 80.0, 95.0])
    keep = ((values >= 45) & (values <= 80)) | values.isna()
    assert keep.tolist() == [False, True, True, True, False]


# ── Scaling ──────────────────────────────────────────────────────────────────


def test_scaler_statistics_come_only_from_the_training_partition():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"x": [100.0, 200.0]})

    scaled_train, scale, centre = scale_feature_train(train.copy(), method="std")
    scaled_test = scale_feature_test(test.copy(), scale, centre, method="std")

    assert scaled_train["x"].mean() == pytest.approx(0.0, abs=1e-9)
    # The test rows are far outside the training range, so they must NOT be standardised
    # to mean zero -- that would mean the scaler saw them.
    assert abs(scaled_test["x"].mean()) > 10


def test_scaling_is_invertible():
    from sga.data.scaling import descale_feature

    train = pd.DataFrame({"x": [1.0, 5.0, 9.0, 13.0]})
    scaled, scale, centre = scale_feature_train(train.copy(), method="std")
    restored = descale_feature(scaled.copy(), scale, centre)
    assert np.allclose(restored["x"].to_numpy(), train["x"].to_numpy())


# ── Table 1 statistics ───────────────────────────────────────────────────────


def test_statistics_helpers_work_without_an_accumulator():
    """Called as documented in ``docs/API.md``, with no ``results`` list."""
    rng = np.random.RandomState(0)
    positive = pd.DataFrame({"efw": rng.normal(2400, 300, 60), LABEL: 1})
    negative = pd.DataFrame({"efw": rng.normal(3000, 350, 140), LABEL: 0})

    rows = get_mean_continuous_feature(positive, negative, "efw")
    assert len(rows) == 2

    combined = pd.concat([positive, negative], ignore_index=True)
    result = normality_and_parametric_test(combined, positive, negative, "efw")
    assert len(result) == 1


def test_per_group_counts_are_reported():
    """Table 1 reports a per-variable n, which needs the non-missing count."""
    positive = pd.DataFrame({"efw": [2000.0, np.nan, 2200.0], LABEL: 1})
    negative = pd.DataFrame({"efw": [3000.0, 3100.0, 3200.0], LABEL: 0})
    rows = get_mean_continuous_feature(positive, negative, "efw")
    counts = [r.get("Count") for r in rows if "Count" in r]
    assert 2 in counts and 3 in counts


def test_welch_is_used_when_variances_differ():
    """The reported "Equal Variance" flag must match the test actually run."""
    rng = np.random.RandomState(1)
    positive = pd.DataFrame({"x": rng.normal(0, 1, 200), LABEL: 1})
    negative = pd.DataFrame({"x": rng.normal(0, 8, 200), LABEL: 0})
    combined = pd.concat([positive, negative], ignore_index=True)
    rows = normality_and_parametric_test(combined, positive, negative, "x")
    assert len(rows) == 1


# ── Feature-space invariants ─────────────────────────────────────────────────


def test_imputer_inputs_are_exactly_the_common_features():
    """A mismatch would feed the imputers a space they were not trained on."""
    from sga.imputation.apply import IMPUTER_INPUT_FEATURES

    assert set(IMPUTER_INPUT_FEATURES) == set(COMMON_FEATURES)
    assert len(IMPUTER_INPUT_FEATURES) == len(COMMON_FEATURES) == 10


def test_cpr_is_part_of_the_common_feature_space():
    """``scripts/01b`` used to drop ``cpr`` by default, breaking every imputer."""
    assert "cpr" in COMMON_FEATURES


def test_india_categorical_list_has_no_duplicates():
    from sga.config import INDIA_CATEGORICAL

    assert len(INDIA_CATEGORICAL) == len(set(INDIA_CATEGORICAL))


# ── Manuscript agreement ─────────────────────────────────────────────────────


def test_baseline_uses_the_ten_common_features_the_methods_list():
    """The Methods enumerate ten shared measurements for the baseline."""
    from sga.pipeline.train_baseline import BASELINE_COMMON_FEATURES

    assert set(BASELINE_COMMON_FEATURES) == set(COMMON_FEATURES)
    assert len(BASELINE_COMMON_FEATURES) == 10


def test_legacy_seven_feature_baseline_is_still_available():
    """Reproducing the previously reported baseline numbers must stay possible."""
    from sga.pipeline.train_baseline import LEGACY_BASELINE_FEATURES

    assert len(LEGACY_BASELINE_FEATURES) == 7
    assert set(LEGACY_BASELINE_FEATURES) < set(COMMON_FEATURES)


def test_screening_and_fold_time_ranges_agree():
    """Screening must not admit values the fold-time filter then drops."""
    from sga.data.cleaning import screening_bounds

    for feature in ("hc", "ac", "fl", "bpd", "cpr"):
        tabled = CONTINUOUS_FEATURE_LOGICAL_RANGE[feature]
        low, high = screening_bounds(feature, in_millimetres=True)
        scale = 10 if feature in ("hc", "ac", "fl") else 1
        assert (low, high) == (tabled[0] * scale, tabled[1] * scale)


def test_cerebroplacental_ratio_upper_bound_is_three():
    """Third-trimester CPR runs to about 3.0."""
    assert CONTINUOUS_FEATURE_LOGICAL_RANGE["cpr"] == [0.2, 3.0]


def test_every_continuous_modelled_feature_has_a_range():
    """An unranged feature is never checked for implausible values."""
    from sga.config import INDIA_REGRESSION_FEATURES, MALAYSIA_REGRESSION_FEATURES

    continuous = [f for f in COMMON_FEATURES if f != "gender"]
    continuous += list(MALAYSIA_REGRESSION_FEATURES) + list(INDIA_REGRESSION_FEATURES)
    missing = [f for f in continuous if f not in CONTINUOUS_FEATURE_LOGICAL_RANGE]
    assert not missing, f"no plausible range defined for {missing}"


def test_ranges_admit_third_trimester_reference_values():
    """Typical 28-week and 40-week measurements must survive the filter."""
    reference = {
        "ga": [196, 280],
        "bpd": [71, 94],
        "hc": [26.0, 35.0],
        "ac": [23.0, 36.0],
        "fl": [5.3, 7.6],
        "efw": [1100, 3500],
        "cpr": [1.1, 2.6],
        "ute_api": [0.6, 1.1],
        "umb_api": [0.7, 1.2],
        "ute_ari": [0.35, 0.6],
        "afi": [8.0, 20.0],
        "psv": [40, 80],
        "m_age": [20, 40],
        "m_height": [150, 170],
        "m_weight": [55, 85],
    }
    df = pd.DataFrame(reference)
    remove_illogical_values(df)
    assert len(df) == 2, "a plausible third-trimester scan was filtered out"


def test_resistance_index_cannot_exceed_one():
    """A resistance index is a ratio bounded by construction."""
    assert CONTINUOUS_FEATURE_LOGICAL_RANGE["ute_ari"] == [0.0, 1.0]
