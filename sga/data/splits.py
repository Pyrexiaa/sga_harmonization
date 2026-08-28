"""Patient-grouped stratified k-fold assignment."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from sga.config import EXTERNAL_TEST_FOLD, LABEL, N_FOLDS_TOTAL, SEED
from sga.data.cleaning import iterative_impute
from sga.data.encoding import convert_feature_to_one_hot


def assign_folds(
    df, num_folds=N_FOLDS_TOTAL, label=LABEL, id_exist=True, seed=SEED, keep_id=True
):
    """Add a ``fold`` column using stratified (grouped) k-fold assignment."""
    df["fold"] = -1
    if id_exist:
        splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        folds = splitter.split(df, df[label], groups=df["id"])
    else:
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        folds = splitter.split(df, df[label])

    fold_position = df.columns.get_loc("fold")
    for fold_num, (_, val_idx) in enumerate(folds):
        # Positional indexing:
        df.iloc[val_idx, fold_position] = fold_num

    if id_exist and not keep_id:
        return df.drop(columns="id")
    return df


def _one_hot_block(df, categorical_col):
    """One-hot encode ``categorical_col`` on the non-missing rows of ``df``."""
    names = []
    for col in categorical_col:
        complete = df[[col]].dropna()
        complete, cols = convert_feature_to_one_hot(complete, col)
        df = pd.concat([df.drop(columns=col), complete], axis=1)
        names.extend(cols)
    return df, names


def split_complete_and_addon(
    df,
    categorical_col=None,
    num_folds=N_FOLDS_TOTAL,
    label=LABEL,
    id_exist=True,
    impute_data=False,
    seed=SEED,
    external_fold=EXTERNAL_TEST_FOLD,
):
    """Split a cohort into complete-case and add-on partitions and assign folds.

    ``impute_data`` defaults to False: the add-on records keep their missing values
    and are completed inside each cross-validation fold, by an imputer fitted on
    that fold's training rows only (``dataset.impute_within_feature``). Imputing
    here instead would fit one imputer on the pooled development block, so the rows
    later held out as a validation fold would have shaped the model that filled
    that fold's own training data - the leakage Methods step (2) rules out.

    Keeping the missing values also means the prepared tables reflect what was
    actually measured, which is what Table 1's per-variable n counts describe.

    Set ``impute_data=True`` only to reproduce the older preparation behaviour.
    """
    complete = df.dropna().reset_index(drop=True)
    add_on = df[df.isna().any(axis=1)].dropna(how="all")
    if "gender" in add_on.columns:
        add_on = add_on[add_on["gender"] != 2]
    add_on = add_on.reset_index(drop=True)

    all_nan = add_on.columns[add_on.isna().all(axis=0)]
    add_on = add_on.drop(columns=all_nan)
    complete = complete.drop(columns=[c for c in all_nan if c in complete.columns])

    names = []
    if categorical_col:
        add_on, add_on_names = _one_hot_block(add_on, categorical_col)
        names.extend(add_on_names)
    if categorical_col:
        complete, complete_names = _one_hot_block(complete, categorical_col)
        names.extend(complete_names)

    # Align the indicator columns produced independently on the two partitions.
    for col in set(names):
        add_on[col] = add_on.get(col, 0)
        complete[col] = complete.get(col, 0)

    add_on["fold"] = -1
    complete = assign_folds(
        complete, num_folds=num_folds, label=label, id_exist=id_exist, seed=seed
    )

    if id_exist and "id" in add_on.columns:
        external = int(
            add_on["id"]
            .isin(set(complete.loc[complete["fold"] == external_fold, "id"].unique()))
            .sum()
        )
        if external:
            print(
                f"{external} add-on scan(s) belong to fold-{external_fold} "
                "pregnancies; they are retained here so the written tables carry the "
                "full cohort, and excluded from every development run by "
                "`dataset.load_cohort(exclude_external_fold=True)`."
            )

    if impute_data and len(add_on):
        fold_column = add_on["fold"]
        id_column = add_on["id"] if "id" in add_on.columns else None
        imputable = add_on.drop(columns=[c for c in ("fold", "id") if c in add_on.columns])
        imputed = iterative_impute(imputable, random_state=seed)
        imputed["fold"] = fold_column.to_numpy()
        if id_column is not None:
            imputed["id"] = id_column.to_numpy()
        add_on = imputed[[c for c in add_on.columns if c in imputed.columns]]

    return complete, add_on
