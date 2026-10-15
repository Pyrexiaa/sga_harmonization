"""Categorical encoding helpers used by the country preprocessing scripts."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def is_numeric(value) -> bool:
    """True when ``value`` can be parsed as a float."""
    try:
        float(value)
    except (ValueError, TypeError):
        return False
    return True


def convert_feature_to_label(df, col):
    """Integer-encode one categorical column in place."""
    df[col] = LabelEncoder().fit_transform(df[col])
    return df


def convert_feature_to_one_hot(df, col):
    """One-hot encode ``col`` into ``<col>_<level>`` indicator columns."""
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

    dummies = pd.DataFrame(
        OneHotEncoder(handle_unknown="error").fit_transform(df[[col]]).toarray()
    )
    df = df.join(dummies)

    rename = {i: f"{col}_{int(name)}" for i, name in zip(dummies.columns, encoder.classes_)}
    df = df.rename(columns=rename).drop(columns=col)
    return df, list(rename.values())


def one_hot(df, categorical_col):
    """One-hot encode several columns and move the indicators to the front."""
    indicators = []
    for col in categorical_col:
        df, names = convert_feature_to_one_hot(df, col)
        indicators.extend(names)

    remaining = [c for c in df.columns if c not in indicators]
    return df[indicators + remaining], len(indicators)


def encode_india_categoricals(df):
    """Apply the India-cohort encoding scheme used during preprocessing."""
    for col in ["hypertension", "diabetes"]:
        df, _ = convert_feature_to_one_hot(df, col)
    for col in ["gender", "smoking"]:
        df = convert_feature_to_label(df, col)
    return df.replace({"oligohydramnios": 0, "normal": 1, "polyhydramnios": 2})
