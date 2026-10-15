"""Shared fixtures: a synthetic two-cohort dataset in the real schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sga.config import (
    COMMON_FEATURES,
    INDIA_BINARY_FEATURES,
    INDIA_REGRESSION_FEATURES,
    LABEL,
    MALAYSIA_MULTICLASS_FEATURES,
    MALAYSIA_REGRESSION_FEATURES,
)
from sga.data.cleaning import CONTINUOUS_FEATURE_LOGICAL_RANGE
from sga.data.splits import assign_folds


def _in_range(rng, n, column):
    """Draw ``n`` plausible values for ``column``, safely inside its valid range."""
    low, high = CONTINUOUS_FEATURE_LOGICAL_RANGE.get(column, (0.0, 1.0))
    margin = 0.15 * (high - low)
    return rng.uniform(low + margin, high - margin, n)


def _cohort(rng, n, continuous, binary, multiclass, with_id):
    data = {c: _in_range(rng, n, c) for c in COMMON_FEATURES if c != "gender"}
    data["gender"] = rng.randint(0, 2, n)
    for column in continuous:
        data[column] = _in_range(rng, n, column)
    for column in binary:
        data[column] = rng.randint(0, 2, n)
    for column in multiclass:
        data[column] = rng.randint(0, 3, n)

    df = pd.DataFrame(data)
    z = (df["efw"] - df["efw"].mean()) / df["efw"].std()
    df[LABEL] = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-2.1 - 1.3 * z)))).astype(int)
    if with_id:
        # Two scans per pregnancy, so the grouped split has something to group.
        df["id"] = np.repeat(np.arange((n + 1) // 2), 2)[:n]
    return df


@pytest.fixture(scope="session")
def synthetic_cohorts():
    """``(malaysia_pair, india_pair)``, each ``[complete_df, add_on_df]``."""
    rng = np.random.RandomState(7)
    malaysia = _cohort(
        rng, 400, MALAYSIA_REGRESSION_FEATURES, [], MALAYSIA_MULTICLASS_FEATURES, True
    )
    india = _cohort(rng, 160, INDIA_REGRESSION_FEATURES, INDIA_BINARY_FEATURES, [], False)
    for column in ("smoking", "diabetes_1", "hypertension_1"):
        india[column] = 0

    pairs = []
    for df, id_exists in ((malaysia, True), (india, False)):
        complete = assign_folds(df.copy(), id_exist=id_exists)
        add_on = complete.sample(frac=0.15, random_state=1).copy()
        add_on["fold"] = -1
        pairs.append([complete.reset_index(drop=True), add_on.reset_index(drop=True)])
    return tuple(pairs)


@pytest.fixture
def prepared_cohort_dir(tmp_path, synthetic_cohorts):
    """Write the synthetic cohorts to disk in the layout ``load_cohort`` expects."""
    (malaysia, india) = synthetic_cohorts
    for name, pair in (("Malaysia", malaysia), ("India", india)):
        out = tmp_path / name
        out.mkdir(parents=True, exist_ok=True)
        pair[0].to_csv(out / "tri3_i21.csv", index=False)
        pair[1].to_csv(out / "tri3_add_on_i21.csv", index=False)
    return tmp_path
