"""The fold builder must hand back cohort labels and pregnancy keys that line up.

The pregnancy-cluster bootstrap in the manuscript resamples whole pregnancies. If
the cluster key is recovered by re-slicing the raw frame afterwards, a single row
dropped by the physiological filter shifts every key by one and the interval is
quietly wrong. Carrying it on a marker column through the filter, as
``prepare_fold`` does, is what these tests pin down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sga.config import COMMON_FEATURES, INDIA, LABEL, MALAYSIA
from sga.data.cleaning import CONTINUOUS_FEATURE_LOGICAL_RANGE as RANGES
from sga.data.splits import assign_folds
from sga.pipeline.harmonized_fold import prepare_fold


def _cohort(rng, n, with_id, scans_per_pregnancy=1):
    data = {}
    for column in COMMON_FEATURES:
        if column == "gender":
            data[column] = rng.randint(0, 2, n)
        else:
            low, high = RANGES[column]
            margin = 0.2 * (high - low)
            data[column] = rng.uniform(low + margin, high - margin, n)
    frame = pd.DataFrame(data)
    frame[LABEL] = rng.randint(0, 2, n)
    if with_id:
        frame["id"] = np.repeat(
            np.arange((n + scans_per_pregnancy - 1) // scans_per_pregnancy),
            scans_per_pregnancy,
        )[:n]
    return frame


@pytest.fixture
def repeated_scan_fold():
    """A fold whose Malaysian cohort has two scans per pregnancy."""
    rng = np.random.RandomState(11)
    malaysia = assign_folds(_cohort(rng, 300, True, scans_per_pregnancy=2), id_exist=True)
    india = assign_folds(_cohort(rng, 120, False), id_exist=False)
    empty = malaysia.iloc[0:0]
    return prepare_fold(
        [malaysia, empty],
        [india, india.iloc[0:0]],
        fold=0,
        selected_features=(),
        train_source="both",
    )


def test_cluster_ids_are_returned_and_aligned(repeated_scan_fold):
    fold = repeated_scan_fold
    assert fold.cluster_ids is not None
    assert len(fold.cluster_ids) == len(fold.test_X) == len(fold.country_arr)
    assert "__cluster__" not in fold.features


def test_cluster_ids_never_span_cohorts(repeated_scan_fold):
    """A pregnancy belongs to exactly one cohort, so no key may appear in both."""
    fold = repeated_scan_fold
    malaysian = set(fold.cluster_ids[fold.country_arr == MALAYSIA])
    indian = set(fold.cluster_ids[fold.country_arr == INDIA])
    assert malaysian and indian
    assert not (malaysian & indian)


def test_each_indian_record_is_its_own_pregnancy(repeated_scan_fold):
    fold = repeated_scan_fold
    indian = fold.cluster_ids[fold.country_arr == INDIA]
    assert len(set(indian)) == len(indian)


def test_repeated_malaysian_scans_share_a_cluster(repeated_scan_fold):
    """Otherwise the cluster bootstrap silently degenerates to a scan bootstrap."""
    fold = repeated_scan_fold
    malaysian = fold.cluster_ids[fold.country_arr == MALAYSIA]
    assert len(set(malaysian)) < len(malaysian)


def test_cluster_ids_survive_the_physiological_filter():
    """A dropped test row must shift the cohort labels and the keys together."""
    rng = np.random.RandomState(5)
    malaysia = assign_folds(_cohort(rng, 300, True), id_exist=True)
    india = _cohort(rng, 120, False)
    india["cpr"] = 1.0
    india.loc[0, "cpr"] = 3.5  # passes India screening, fails the fold filter
    india = assign_folds(india, id_exist=False)
    india.loc[0, "fold"] = 0

    fold = prepare_fold(
        [malaysia, malaysia.iloc[0:0]],
        [india, india.iloc[0:0]],
        fold=0,
        selected_features=(),
        train_source="both",
    )
    assert fold.cluster_ids is not None
    assert len(fold.cluster_ids) == len(fold.test_X)
    # Every surviving Indian row must still carry an Indian key.
    indian_keys = fold.cluster_ids[fold.country_arr == INDIA]
    assert all(str(key).startswith("IN:") for key in indian_keys)
    malaysian_keys = fold.cluster_ids[fold.country_arr == MALAYSIA]
    assert all(str(key).startswith("MY:") for key in malaysian_keys)
