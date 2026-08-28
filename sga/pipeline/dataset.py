"""Cohort loading and per-fold train/test construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sga.config import (
    CHART,
    EXTERNAL_TEST_FOLD,
    INDIA_BINARY_FEATURES,
    INDIA_SUBDIR,
    LABEL,
    MALAYSIA_CATEGORICAL,
    MALAYSIA_SUBDIR,
    PREV_PREGNANCY_FEATURES,
    SEED,
    TRAINING_DATA_DIR,
)
from sga.data.cleaning import apply_iterative_imputer, fit_iterative_imputer
from sga.data.scaling import scale_feature_test, scale_feature_train

# Columns treated as categorical when the India feature space is in play.
_INDIA_CATEGORICAL = sorted(
    set(INDIA_BINARY_FEATURES) | set(PREV_PREGNANCY_FEATURES) | {"gender"}
)


def load_cohort(subdir, chart=CHART, base_dir=None, exclude_external_fold=True,
                external_fold=EXTERNAL_TEST_FOLD):
    """Load one cohort as ``[complete_df, add_on_df]``."""
    base = base_dir or TRAINING_DATA_DIR
    main = pd.read_csv(f"{base}/{subdir}/tri3_{chart}.csv")
    add_on = pd.read_csv(f"{base}/{subdir}/tri3_add_on_{chart}.csv")

    if exclude_external_fold:
        external = main["fold"] == external_fold
        if "id" in main.columns and "id" in add_on.columns:
            external_ids = set(main.loc[external, "id"].unique())
            add_on = add_on[~add_on["id"].isin(external_ids)]
        main = main[~external]

    return [main.reset_index(drop=True), add_on.reset_index(drop=True)]


def load_both_cohorts(chart=CHART, base_dir=None, exclude_external_fold=True):
    """Load Malaysia and India as a ``(malaysia, india)`` pair of dataset pairs."""
    return (
        load_cohort(MALAYSIA_SUBDIR, chart, base_dir, exclude_external_fold),
        load_cohort(INDIA_SUBDIR, chart, base_dir, exclude_external_fold),
    )


def separate_df_and_df_add_on(dfs, label=LABEL, id_exists=True, additional_drop_columns=None):
    """Align a ``[complete, add_on]`` pair and classify features by type."""
    df, df_add_on = dfs[0], dfs[1][dfs[0].columns]

    drop_columns = [label, "fold"] + (["id"] if id_exists else [])
    if additional_drop_columns:
        if not isinstance(additional_drop_columns, list):
            raise TypeError("additional_drop_columns must be a list")
        drop_columns += additional_drop_columns

    features = df.columns.astype(str).drop(drop_columns).tolist()
    known_categorical = MALAYSIA_CATEGORICAL if id_exists else _INDIA_CATEGORICAL
    continuous_features = [f for f in features if f not in known_categorical]
    categorical_features = [f for f in features if f in known_categorical]

    df = df[categorical_features + continuous_features + drop_columns].copy()
    df_add_on = df_add_on[df.columns].copy()

    for frame in (df, df_add_on):
        frame[categorical_features] = frame[categorical_features].astype(int)
        frame[label] = frame[label].astype(int)

    if additional_drop_columns:
        df = df.drop(columns=additional_drop_columns)
        df_add_on = df_add_on.drop(columns=additional_drop_columns)

    return df, df_add_on, categorical_features, continuous_features, features


def impute_within_feature(train_df, test_df=None, label=LABEL, seed=SEED, verbose=False):
    """Complete within-feature missing values, fitting on the TRAINING rows only.

    Methods step (2): "remaining within-feature missing values were completed by
    iterative imputation fitted on the training partition". Fitting inside the fold
    is what makes that true. Fitting once on the pooled development block instead
    lets the rows later held out as fold 3's validation set shape the imputer that
    fills fold 3's own training rows.

    The outcome column is excluded from the imputation model: it is available for
    training rows but not for the rows a deployed model would score, so letting it
    inform the feature values would be an outcome leak.

    Returns ``(train_df, test_df, imputer)``; ``imputer`` is ``None`` when nothing
    was missing, so callers can pass it on to another partition (the external test
    fold is scored with the imputer of the fold-model that scores it).
    """
    feature_columns = [c for c in train_df.columns if c != label]
    frames = [f for f in (train_df, test_df) if f is not None]
    if not feature_columns or not any(
        f[feature_columns].isna().any().any() for f in frames
    ):
        return train_df, test_df, None

    imputer = fit_iterative_imputer(
        train_df[feature_columns], random_state=seed, verbose=verbose
    )
    train_df = train_df.copy()
    train_df[feature_columns] = apply_iterative_imputer(
        imputer, train_df[feature_columns]
    )
    if test_df is not None:
        test_df = test_df.copy()
        test_df[feature_columns] = apply_iterative_imputer(
            imputer, test_df[feature_columns]
        )
    if verbose:
        print(f"    within-feature imputer fitted on {len(train_df)} training rows")
    return train_df, test_df, imputer


def apply_within_feature_imputer(imputer, df, label=LABEL):
    """Apply a fold's fitted within-feature imputer to another partition."""
    if imputer is None:
        return df
    feature_columns = [c for c in df.columns if c != label]
    df = df.copy()
    df[feature_columns] = apply_iterative_imputer(imputer, df[feature_columns])
    return df


def process_raw_train_and_test_df(
    df,
    df_add_on,
    fold,
    id_exists=True,
    add_noise_features=None,
    seed=SEED,
    label=LABEL,
    impute_within=True,
    return_imputer=False,
):
    """Build the train/test partition for ``fold``.

    Add-on (incomplete) records join the TRAINING rows only, and never those whose
    pregnancy is in this fold's test partition. Any missing values they still carry
    are completed by an iterative imputer fitted on these training rows alone - see
    :func:`impute_within_feature`. When the prepared tables were written by an older
    run that imputed at preparation time there is nothing missing, the step is
    skipped, and the returned imputer is ``None``.

    With ``return_imputer=True`` the fitted imputer is returned as a third value so
    the same one can be applied to another partition.
    """
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    test_df = df[df["fold"] == fold].reset_index(drop=True)

    if id_exists:
        merged = pd.merge(df_add_on, test_df["id"], on="id", how="left", indicator=True)
        add_on_train = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
        train_df = pd.concat([add_on_train, train_df], ignore_index=True)
        train_df = train_df.drop(columns=["fold", "id"])
        test_df = test_df.drop(columns=["fold", "id"])
    else:
        train_df = pd.concat([train_df, df_add_on], axis=0, ignore_index=True)
        train_df = train_df.drop(columns="fold")
        test_df = test_df.drop(columns="fold")

    imputer = None
    if impute_within:
        train_df, test_df, imputer = impute_within_feature(
            train_df, test_df, label=label, seed=seed
        )

    if add_noise_features:
        rng = np.random.RandomState(seed + fold)
        for feature in add_noise_features:
            noise = rng.normal(0, train_df[feature].std() / 5, len(train_df))
            train_df[feature] += noise

    if return_imputer:
        return train_df, test_df, imputer
    return train_df, test_df


def scale_sample_train_and_test_df(
    train_df,
    test_df,
    validation_df=None,
    categorical_features=(),
    continuous_features=(),
    test_on_other_csv=None,
    sample_size=None,
    seed=SEED,
):
    """Standardise continuous features using training statistics only."""
    continuous_features = list(continuous_features)
    train_df[continuous_features], std_or_min, mean_or_max = scale_feature_train(
        train_df[continuous_features], method="std"
    )
    for feature in categorical_features:
        train_df[feature] = train_df[feature].astype(int)

    if test_on_other_csv is not None:
        test_df = test_on_other_csv.drop(columns=["fold"])
    test_df[continuous_features] = scale_feature_test(
        test_df[continuous_features], std_or_min, mean_or_max, method="std"
    )
    if validation_df is not None:
        validation_df[continuous_features] = scale_feature_test(
            validation_df[continuous_features], std_or_min, mean_or_max, method="std"
        )

    if sample_size is not None:
        train_df = train_df.sample(n=sample_size, random_state=seed).copy()

    ordered = [c for c in train_df.columns if c not in continuous_features] + continuous_features
    train_df = train_df[ordered]
    test_df = test_df[ordered]
    if validation_df is not None:
        validation_df = validation_df[ordered]

    return train_df, test_df, validation_df, std_or_min, mean_or_max


def scale_training_partition(train_df, scale_factor, label=LABEL, random_state=None):
    """Subsample a TRAINING partition to ``scale_factor`` of its size."""
    if scale_factor is None or scale_factor >= 1.0:
        return train_df.reset_index(drop=True)

    rng = np.random.RandomState(random_state)
    positive, negative = train_df[train_df[label] == 1], train_df[train_df[label] == 0]
    n_pos = max(1, int(round(len(positive) * scale_factor))) if len(positive) else 0
    n_neg = max(1, int(round(len(negative) * scale_factor))) if len(negative) else 0

    parts = []
    if n_pos:
        parts.append(positive.sample(n=min(n_pos, len(positive)), random_state=rng))
    if n_neg:
        parts.append(negative.sample(n=min(n_neg, len(negative)), random_state=rng))

    scaled = pd.concat(parts) if parts else train_df.iloc[0:0]
    return scaled.sample(frac=1, random_state=rng).reset_index(drop=True)


def stratified_subsample(df, n, label=LABEL, seed=None):
    """Subsample ``df`` to ``n`` rows without replacement, preserving class ratio."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)

    rng = np.random.RandomState(seed)
    positive, negative = df[df[label] == 1], df[df[label] == 0]
    fraction = n / len(df)
    n_pos = max(1, int(round(len(positive) * fraction))) if len(positive) else 0
    n_neg = max(1, n - n_pos) if len(negative) else 0

    parts = []
    if n_pos:
        parts.append(positive.sample(n=min(n_pos, len(positive)), random_state=rng))
    if n_neg:
        parts.append(negative.sample(n=min(n_neg, len(negative)), random_state=rng))

    out = pd.concat(parts) if parts else df.iloc[0:0]
    return out.sample(frac=1, random_state=rng).reset_index(drop=True)


def cast_common_types(df):
    """Normalise dtypes shared by both cohorts before imputation."""
    df["m_age"] = df["m_age"].astype(float)
    df["gender"] = df["gender"].astype(int)
    df.reset_index(drop=True, inplace=True)
    return df
