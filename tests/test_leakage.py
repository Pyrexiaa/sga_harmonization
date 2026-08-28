"""Regression tests for the leakage guarantees stated in ``docs/PIPELINE.md``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sga.config import EXTERNAL_TEST_FOLD, LABEL, SEED
from sga.data.splits import assign_folds, split_complete_and_addon
from sga.pipeline.dataset import load_cohort, process_raw_train_and_test_df


def test_load_cohort_drops_external_fold_from_both_frames(prepared_cohort_dir):
    """The external fold must leave the complete-case AND the add-on frame."""
    raw = pd.read_csv(prepared_cohort_dir / "Malaysia" / "tri3_i21.csv")
    raw_add_on = pd.read_csv(prepared_cohort_dir / "Malaysia" / "tri3_add_on_i21.csv")
    external_ids = set(raw.loc[raw["fold"] == EXTERNAL_TEST_FOLD, "id"])

    # The fixture must actually contain the hazard, or the test proves nothing.
    assert raw_add_on["id"].isin(external_ids).sum() > 0

    main, add_on = load_cohort("Malaysia", base_dir=prepared_cohort_dir)

    assert (main["fold"] != EXTERNAL_TEST_FOLD).all()
    assert add_on["id"].isin(external_ids).sum() == 0


def test_load_cohort_keeps_external_fold_when_asked(prepared_cohort_dir):
    """The external-test scripts need the unfiltered frames."""
    main, _ = load_cohort("Malaysia", base_dir=prepared_cohort_dir, exclude_external_fold=False)
    assert (main["fold"] == EXTERNAL_TEST_FOLD).any()


def test_cv_fold_training_partition_excludes_test_pregnancies(prepared_cohort_dir):
    """No pregnancy may appear in both sides of a cross-validation split."""
    main, add_on = load_cohort("Malaysia", base_dir=prepared_cohort_dir)
    for fold in range(4):
        train_ids = set(main.loc[main["fold"] != fold, "id"])
        test_ids = set(main.loc[main["fold"] == fold, "id"])
        assert not (train_ids & test_ids), f"fold {fold} shares pregnancies"

        train_df, test_df = process_raw_train_and_test_df(
            main.copy(), add_on.copy(), fold, id_exists=True
        )
        assert len(train_df) > 0 and len(test_df) > 0


def test_addon_from_external_pregnancies_never_reaches_a_development_run():
    """Fold-4 pregnancies must not leak in through their add-on scans.

    The scans are RETAINED when the tables are written, so the prepared cohort
    still holds every record Table 1 counts; the exclusion happens at load time,
    which is the only place that knows whether a development or an external run is
    being set up.
    """
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n // 2), 2),
            "efw": rng.uniform(1500, 3500, n),
            "hc": rng.uniform(25, 40, n),
            "ac": rng.uniform(22, 38, n),
            LABEL: rng.randint(0, 2, n),
        }
    )
    incomplete = df.sample(frac=0.5, random_state=1).index
    df.loc[incomplete[: len(incomplete) // 2], "hc"] = np.nan
    df.loc[incomplete[len(incomplete) // 2 :], "ac"] = np.nan

    complete, add_on = split_complete_and_addon(df.copy(), id_exist=True)
    external_ids = set(complete.loc[complete["fold"] == EXTERNAL_TEST_FOLD, "id"])

    # Nothing is discarded at preparation time...
    assert len(complete) + len(add_on) == len(df)
    assert add_on["id"].dtype.kind in "iu", "id must not be coerced to float"
    # ...and the missing values survive, to be filled inside each fold instead.
    assert add_on[["hc", "ac"]].isna().any().any()

    # ...but a development load drops every add-on scan of an external pregnancy.
    kept = add_on[~add_on["id"].isin(external_ids)]
    assert len(kept) < len(add_on)
    assert not kept["id"].isin(external_ids).any()


def test_within_feature_imputer_is_fitted_on_training_rows_only():
    """A held-out row must not be able to change what the training rows are filled with.

    The check is behavioural rather than structural: the held-out fold is replaced
    by wildly out-of-scale values and the training imputations must come out
    bit-identical. They only can if the imputer never saw those rows.
    """
    from sga.pipeline.dataset import process_raw_train_and_test_df

    rng = np.random.RandomState(3)
    n = 300
    base = pd.DataFrame(
        {
            "id": np.arange(n),
            "efw": rng.uniform(1500, 3500, n),
            "hc": rng.uniform(25, 40, n),
            "ac": rng.uniform(22, 38, n),
            LABEL: rng.randint(0, 2, n),
        }
    )
    base["fold"] = np.tile(np.arange(5), n // 5)
    base.loc[base.sample(frac=0.3, random_state=2).index, "hc"] = np.nan

    add_on = base[base["hc"].isna()].copy()
    add_on["fold"] = -1
    complete = base[base["hc"].notna()].copy()

    def train_rows(test_fold_values):
        frame = complete.copy()
        frame.loc[frame["fold"] == 0, "ac"] = test_fold_values
        train_df, _ = process_raw_train_and_test_df(frame, add_on, 0, id_exists=True)
        return train_df.sort_values(["efw", "ac"]).reset_index(drop=True)

    unchanged = train_rows(complete.loc[complete["fold"] == 0, "ac"].to_numpy())
    corrupted = train_rows(np.full((complete["fold"] == 0).sum(), 1e6))

    pd.testing.assert_frame_equal(unchanged, corrupted)
    assert unchanged["hc"].notna().all(), "training gaps should still be filled"


def test_assign_folds_is_positional_not_label_based():
    """Fold assignment must not depend on the frame's index labels."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(60), 2),
            LABEL: rng.randint(0, 2, 120),
            "x": rng.normal(size=120),
        }
    )
    shifted = df.copy()
    shifted.index = np.arange(1000, 1120)

    clean = assign_folds(df.copy())
    relabelled = assign_folds(shifted.copy())

    assert (clean["fold"] == -1).sum() == 0
    assert (relabelled["fold"] == -1).sum() == 0
    assert relabelled.groupby("id")["fold"].nunique().max() == 1


def test_per_fold_imputers_never_see_the_held_out_rows(synthetic_cohorts):
    """A refitted imputer must differ from one fitted on a different fold."""
    from sga.imputation.fold_imputers import fit_fold_imputers
    from sga.pipeline.dataset import separate_df_and_df_add_on

    malaysia, india = synthetic_cohorts
    msia_df, msia_add, *_ = separate_df_and_df_add_on(malaysia, LABEL, id_exists=True)
    india_df, india_add, *_ = separate_df_and_df_add_on(india, LABEL, id_exists=False)

    predictions = []
    for fold in (0, 1):
        msia_train, _ = process_raw_train_and_test_df(
            msia_df.copy(), msia_add.copy(), fold, id_exists=True
        )
        india_train, india_test = process_raw_train_and_test_df(
            india_df.copy(), india_add.copy(), fold, id_exists=False
        )
        imputers = fit_fold_imputers(
            {"malaysia": msia_train, "india": india_train},
            [("ute_ari", "regression")],
            seed=SEED,
        )
        assert "ute_ari" in imputers, "the fold should support this imputer"

        from sga.imputation.apply import impute_df

        scored = impute_df(
            india_test.drop(columns=[c for c in ["ute_ari"] if c in india_test.columns]),
            ["ute_ari"],
            regression_features=["ute_ari"],
            imputers=imputers,
        )
        predictions.append(scored["ute_ari"].to_numpy()[:20])

    assert not np.allclose(predictions[0], predictions[1]), (
        "fold 0 and fold 1 produced identical imputations, which means one "
        "globally fitted imputer was reused instead of refitting per fold"
    )


def test_fit_fold_imputers_rejects_a_missing_source_cohort(synthetic_cohorts):
    """A silent fallback here would reintroduce the global-imputer bug."""
    from sga.imputation.fold_imputers import fit_fold_imputers

    malaysia, _ = synthetic_cohorts
    with pytest.raises(KeyError, match="not supplied"):
        fit_fold_imputers({"india": malaysia[0]}, [("ute_ari", "regression")])


def test_per_cohort_labels_survive_the_physiological_filter():
    """Dropping a test row must not disable per-cohort evaluation."""
    from sga.config import COMMON_FEATURES
    from sga.data.cleaning import CONTINUOUS_FEATURE_LOGICAL_RANGE as ranges
    from sga.pipeline.harmonized_fold import INDIA, MALAYSIA, prepare_fold

    rng = np.random.RandomState(3)

    def cohort(n, with_id):
        data = {}
        for column in COMMON_FEATURES:
            if column == "gender":
                data[column] = rng.randint(0, 2, n)
            else:
                low, high = ranges[column]
                data[column] = rng.uniform(low + 0.2 * (high - low), high - 0.2 * (high - low), n)
        frame = pd.DataFrame(data)
        frame[LABEL] = rng.randint(0, 2, n)
        if with_id:
            frame["id"] = np.arange(n)
        return frame

    malaysia = assign_folds(cohort(300, True), id_exist=True)
    india = cohort(120, False)
    india["cpr"] = 1.0
    india.loc[0, "cpr"] = 3.5  # passes India screening, fails the fold filter
    india = assign_folds(india, id_exist=False)
    india.loc[0, "fold"] = 0  # put the offending row in the test fold

    empty = lambda frame: frame.iloc[0:0]
    fold = prepare_fold(
        [malaysia, empty(malaysia)],
        [india, empty(india)],
        fold=0,
        selected_features=(),
        train_source="both",
    )

    assert fold.country_arr is not None, "per-cohort evaluation was disabled"
    assert len(fold.country_arr) == len(fold.test_X)
    assert (fold.country_arr == INDIA).sum() > 0
    assert (fold.country_arr == MALAYSIA).sum() > 0
    assert "__country__" not in fold.features
