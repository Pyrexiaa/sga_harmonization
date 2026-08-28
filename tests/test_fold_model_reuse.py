"""The external partition is scored by a development fold-model, not a new one.

Methods 2.3.4: "the best model, judging from the highest AUROC based on the
validation sets, was used to evaluate on the testing data." So fold 4 must be
scored by one of the four fold-models, carrying that fold's own preprocessing -
its within-feature imputer, its cross-domain imputers and its scaler - rather than
by a fresh model refitted on the pooled development block.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sga.config import EXTERNAL_TEST_FOLD
from sga.pipeline.harmonized_fold import prepare_fold


@pytest.fixture
def cohort_pairs(synthetic_cohorts):
    """``(dev, full)`` pairs: the development frames drop the external fold."""
    malaysia, india = synthetic_cohorts

    def development(pair):
        complete, add_on = pair
        external_ids = set(complete.loc[complete["fold"] == EXTERNAL_TEST_FOLD, "id"]) \
            if "id" in complete.columns else set()
        kept_add_on = (
            add_on[~add_on["id"].isin(external_ids)]
            if "id" in add_on.columns
            else add_on
        )
        return [
            complete[complete["fold"] != EXTERNAL_TEST_FOLD].reset_index(drop=True),
            kept_add_on.reset_index(drop=True),
        ]

    return (development(malaysia), development(india)), (malaysia, india)


def test_external_test_fold_keeps_the_development_training_partition(cohort_pairs):
    (msia_dev, india_dev), (msia_full, india_full) = cohort_pairs

    development = prepare_fold(msia_dev, india_dev, 0, selected_features=())
    external = prepare_fold(
        msia_dev,
        india_dev,
        0,
        selected_features=(),
        msia_ds_full=msia_full,
        india_ds_full=india_full,
        external_test_fold=EXTERNAL_TEST_FOLD,
    )

    # Same model, same preprocessing: only the rows being scored change.
    pd.testing.assert_frame_equal(development.train_X, external.train_X)
    assert list(development.features) == list(external.features)
    assert len(external.test_X) != len(development.test_X)


def test_the_external_test_rows_are_the_external_fold(cohort_pairs):
    (msia_dev, india_dev), (msia_full, india_full) = cohort_pairs
    external = prepare_fold(
        msia_dev, india_dev, 0, selected_features=(),
        msia_ds_full=msia_full, india_ds_full=india_full,
        external_test_fold=EXTERNAL_TEST_FOLD,
    )
    expected_malaysia = int(
        (msia_full[0]["fold"] == EXTERNAL_TEST_FOLD).sum()
    )
    assert external.n_msia_test == expected_malaysia
    assert external.country_arr is not None
    assert len(external.country_arr) == len(external.test_X)
    assert not external.test_X.isna().any().any(), (
        "external rows must arrive complete, filled by this fold's own imputer"
    )


def test_every_development_fold_can_score_the_external_partition(cohort_pairs):
    """All four fold-models must be usable, since the best one is chosen by AUROC."""
    (msia_dev, india_dev), (msia_full, india_full) = cohort_pairs
    sizes = set()
    for fold in range(4):
        external = prepare_fold(
            msia_dev, india_dev, fold, selected_features=(),
            msia_ds_full=msia_full, india_ds_full=india_full,
            external_test_fold=EXTERNAL_TEST_FOLD,
        )
        sizes.add(len(external.test_X))
        assert external.test_Y.sum() > 0
    # Every fold-model is scored on the SAME external rows.
    assert len(sizes) == 1, f"external partition changed between folds: {sizes}"


def test_external_test_fold_requires_the_full_cohorts(cohort_pairs):
    (msia_dev, india_dev), _ = cohort_pairs
    with pytest.raises(ValueError, match="msia_ds_full"):
        prepare_fold(
            msia_dev, india_dev, 0, selected_features=(),
            external_test_fold=EXTERNAL_TEST_FOLD,
        )


def test_the_label_never_enters_the_within_feature_imputer():
    """The outcome is available for training rows but not at scoring time."""
    import inspect

    from sga.pipeline.dataset import impute_within_feature

    source = inspect.getsource(impute_within_feature)
    assert 'c != label' in source, (
        "the outcome column must be excluded from the imputation model"
    )
