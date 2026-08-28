"""Single source of truth for building one harmonized cross-validation fold."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

from sga.config import (
    ALL_CROSS_DOMAIN_FEATURES,
    INDIA,
    CATEGORICAL_FEATURES,
    COMMON_FEATURES,
    CONSTANT_ZERO_FEATURES,
    INDIA_BINARY_FEATURES,
    INDIA_REGRESSION_FEATURES,
    LABEL,
    MALAYSIA,
    MALAYSIA_MULTICLASS_FEATURES,
    MALAYSIA_REGRESSION_FEATURES,
    SEED,
)
from sga.data.cleaning import remove_illogical_values
from sga.imputation.apply import impute_df
from sga.imputation.fold_imputers import fit_fold_imputers
from sga.pipeline.dataset import (
    apply_within_feature_imputer,
    cast_common_types,
    process_raw_train_and_test_df,
    scale_sample_train_and_test_df,
    separate_df_and_df_add_on,
    stratified_subsample,
)

# ``MALAYSIA`` and ``INDIA`` now live in ``sga.config`` so the evaluation layer can
# use them without importing the training pipeline; re-exported here because every
# experiment script imports them from this module.
__all__ = ["INDIA", "MALAYSIA", "FoldData", "prepare_fold"]

#: Internal markers carrying the cohort of origin and the pregnancy identifier
#: through the row-dropping physiological filter, so ``country_arr`` and
#: ``cluster_ids`` can never fall out of alignment with the surviving test rows.
_COUNTRY_COLUMN = "__country__"
_CLUSTER_COLUMN = "__cluster__"


@dataclass
class FoldData:
    """Model-ready arrays for one fold."""

    train_X: pd.DataFrame
    train_Y: np.ndarray
    test_X: pd.DataFrame
    test_Y: np.ndarray
    country_arr: np.ndarray | None
    cluster_ids: np.ndarray | None = None
    cats: list = field(default_factory=list)
    features: list = field(default_factory=list)
    n_msia_test: int = 0
    n_india_test: int = 0
    n_train_raw: int = 0

    def __getitem__(self, key):
        # Dict-style access kept for compatibility with the experiment scripts.
        return getattr(self, key)


def _group_selected(selected_features):
    """Split the requested cross-domain features by imputation model type."""
    selected = [f for f in selected_features if f in set(ALL_CROSS_DOMAIN_FEATURES)]
    return (
        selected,
        [f for f in selected if f in MALAYSIA_MULTICLASS_FEATURES],
        [f for f in selected if f in MALAYSIA_REGRESSION_FEATURES],
        [f for f in selected if f in INDIA_BINARY_FEATURES],
        [f for f in selected if f in INDIA_REGRESSION_FEATURES],
    )


def prepare_fold(
    msia_ds,
    india_ds,
    fold,
    selected_features=(),
    train_source="both",
    subsample_n=None,
    subsample_seed=SEED,
    label=LABEL,
    fit_imputers_per_fold=True,
    imputer_seed=SEED,
    msia_ds_full=None,
    india_ds_full=None,
    external_test_fold=None,
):
    """Build one harmonized fold.

    Normally the training partition is every fold but ``fold``, and the test
    partition is ``fold`` itself.

    Pass ``external_test_fold`` (with the full, fold-4-inclusive cohorts in
    ``msia_ds_full`` / ``india_ds_full``) to keep ``fold``'s training partition but
    swap the test partition for the held-out external fold. That is how a single
    development fold-model is scored on the external partition without retraining
    it on the pooled development block: the model, its within-feature imputer, its
    cross-domain imputers and its scaler all stay those of fold ``fold``, exactly
    as Methods 2.3.4 describes ("the best model, judging from the highest AUROC
    based on the validation sets, was used to evaluate on the testing data").
    """
    if external_test_fold is not None and (msia_ds_full is None or india_ds_full is None):
        raise ValueError(
            "external_test_fold requires the full cohorts in msia_ds_full and "
            "india_ds_full; the development frames have fold "
            f"{external_test_fold} removed."
        )

    selected, msia_mc, msia_reg, ind_bin, ind_reg = _group_selected(selected_features)

    msia_df, msia_add, *_ = separate_df_and_df_add_on(msia_ds, label, id_exists=True)
    india_df, india_add, *_ = separate_df_and_df_add_on(india_ds, label, id_exists=False)

    keep_cols = list(COMMON_FEATURES) + selected
    cats_all = list(dict.fromkeys(["gender"] + [c for c in keep_cols if c in CATEGORICAL_FEATURES]))

    msia_train, msia_test, msia_within_imputer = process_raw_train_and_test_df(
        msia_df, msia_add, fold, id_exists=True, label=label, return_imputer=True
    )
    india_train, india_test, india_within_imputer = process_raw_train_and_test_df(
        india_df, india_add, fold, id_exists=False, label=label, return_imputer=True
    )

    # Pregnancy identifiers for the Malaysian test rows, read in the SAME order
    # ``process_raw_train_and_test_df`` builds them (``df[df.fold == fold]``
    # reset), before that function drops the id column. Recovering them here
    # rather than re-slicing the raw frame afterwards is what keeps the cluster
    # bootstrap aligned when the physiological filter drops a row.
    msia_test_ids = msia_df.loc[msia_df["fold"] == fold, "id"].to_numpy()

    if external_test_fold is not None:
        # Keep this fold's TRAINING partition; replace only the rows being scored.
        msia_df_all, msia_add_all, *_ = separate_df_and_df_add_on(
            msia_ds_full, label, id_exists=True
        )
        india_df_all, india_add_all, *_ = separate_df_and_df_add_on(
            india_ds_full, label, id_exists=False
        )
        _, msia_test = process_raw_train_and_test_df(
            msia_df_all, msia_add_all, external_test_fold, id_exists=True, label=label,
            impute_within=False,
        )
        _, india_test = process_raw_train_and_test_df(
            india_df_all, india_add_all, external_test_fold, id_exists=False,
            label=label, impute_within=False,
        )
        # ...and complete it with THIS fold's imputer, not one refitted on the
        # external partition's own neighbours.
        msia_test = apply_within_feature_imputer(msia_within_imputer, msia_test, label)
        india_test = apply_within_feature_imputer(india_within_imputer, india_test, label)
        msia_test_ids = msia_df_all.loc[
            msia_df_all["fold"] == external_test_fold, "id"
        ].to_numpy()

    for frame in (msia_train, india_train, msia_test, india_test):
        cast_common_types(frame)

    # Refit the cross-domain imputers on THIS fold's training partitions before applying
    # them, so a held-out row's harmonized features never come from a model that was
    # trained on that row (docs/PIPELINE.md, step 4).
    fold_imputers = {}
    if fit_imputers_per_fold and selected:
        targets = (
            [(f, "multiclass") for f in msia_mc]
            + [(f, "regression") for f in msia_reg]
            + [(f, "binary") for f in ind_bin]
            + [(f, "regression") for f in ind_reg]
        )
        targets = [(f, kind) for f, kind in targets if f not in CONSTANT_ZERO_FEATURES]
        if targets:
            fold_imputers = fit_fold_imputers(
                {"malaysia": msia_train, "india": india_train},
                targets,
                seed=imputer_seed,
            )

    if msia_mc or msia_reg:
        india_train = impute_df(
            india_train, msia_mc + msia_reg, (), msia_mc, msia_reg,
            imputers=fold_imputers,
        )
        india_test = impute_df(
            india_test, msia_mc + msia_reg, (), msia_mc, msia_reg,
            imputers=fold_imputers,
        )
    if ind_bin or ind_reg:
        msia_train = impute_df(
            msia_train, ind_bin + ind_reg, ind_bin, (), ind_reg,
            imputers=fold_imputers,
        )
        msia_test = impute_df(
            msia_test, ind_bin + ind_reg, ind_bin, (), ind_reg,
            imputers=fold_imputers,
        )

    keep = keep_cols + [label]

    def subset(frame):
        return frame[[c for c in keep if c in frame.columns]].copy()

    msia_train, msia_test = subset(msia_train), subset(msia_test)
    india_train, india_test = subset(india_train), subset(india_test)

    common = sorted(set(msia_train.columns) & set(india_train.columns))
    msia_train, india_train = msia_train[common], india_train[common]
    msia_test = msia_test[[c for c in common if c in msia_test.columns]]
    india_test = india_test[[c for c in common if c in india_test.columns]]

    # Test set:
    n_msia_test, n_india_test = len(msia_test), len(india_test)
    msia_test = msia_test.copy()
    india_test = india_test.copy()
    msia_test[_COUNTRY_COLUMN] = MALAYSIA
    india_test[_COUNTRY_COLUMN] = INDIA
    # Each Indian record is a single pregnancy, so a synthetic per-row identifier
    # is the correct cluster key there.
    if len(msia_test_ids) == len(msia_test):
        msia_test[_CLUSTER_COLUMN] = [f"MY:{i}" for i in msia_test_ids]
    else:
        msia_test[_CLUSTER_COLUMN] = [f"MY:row{i}" for i in range(len(msia_test))]
        print(
            f"  [warn] {len(msia_test_ids)} Malaysian fold-{fold} ids for "
            f"{len(msia_test)} test rows; falling back to per-scan clusters."
        )
    india_test[_CLUSTER_COLUMN] = [f"IN:{i}" for i in range(len(india_test))]
    test_df = pd.concat([msia_test, india_test], axis=0).reset_index(drop=True)
    test_df[label] = test_df[label].astype(int)

    if train_source == "india":
        train_df = india_train.copy()
    elif train_source == "malaysia":
        train_df = msia_train.copy()
    elif train_source == "both":
        train_df = pd.concat([msia_train, india_train], axis=0)
    else:
        raise ValueError(f"train_source must be both/malaysia/india, got {train_source!r}")

    train_df = train_df.reset_index(drop=True)
    train_df[label] = train_df[label].astype(int)
    train_df = stratified_subsample(train_df, subsample_n, label, subsample_seed)
    n_train_raw = int(len(train_df))

    cats = [c for c in cats_all if c in train_df.columns]
    cont = [c for c in train_df.columns if c not in cats and c != label]

    column_index = {c: i for i, c in enumerate(train_df.columns)}
    smote = SMOTENC(
        sampling_strategy="auto",
        categorical_features=[column_index[c] for c in cats if c in column_index],
        random_state=subsample_seed,
    )
    resampled_X, resampled_y = smote.fit_resample(train_df.drop(columns=label), train_df[label])
    train_df = pd.concat([resampled_X, resampled_y], axis=1)

    for c in cats:
        train_df[c] = train_df[c].astype(int)
        test_df[c] = test_df[c].astype(int)

    remove_illogical_values(train_df)
    remove_illogical_values(test_df)

    # Read the surviving rows' cohort labels and cluster keys off the marker
    # columns BEFORE scaling, which reindexes test_df to the training columns and
    # would drop the markers.
    country_arr = test_df.pop(_COUNTRY_COLUMN).to_numpy(dtype=int)
    cluster_ids = test_df.pop(_CLUSTER_COLUMN).to_numpy()

    train_df, test_df, _, _, _ = scale_sample_train_and_test_df(
        train_df, test_df, None, cats, cont
    )
    features = [c for c in train_df.columns if c not in cont] + cont
    features.remove(label)
    for c in cats:
        train_df[c] = train_df[c].astype(int)
        test_df[c] = test_df[c].astype(int)

    return FoldData(
        train_X=train_df[features],
        train_Y=train_df[label].astype(int).values,
        test_X=test_df[features],
        test_Y=test_df[label].astype(int).values,
        country_arr=country_arr,
        cluster_ids=cluster_ids,
        cats=cats,
        features=features,
        n_msia_test=n_msia_test,
        n_india_test=n_india_test,
        n_train_raw=n_train_raw,
    )
