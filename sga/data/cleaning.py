"""Physiological range filtering and within-feature imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from sga.config import SEED

# Inclusive plausible ranges for third-trimester scans (28+0 to 42+6 weeks), wide enough
# to admit genuine pathology and narrow enough to reject keying errors.
CONTINUOUS_FEATURE_LOGICAL_RANGE = {
    # Third trimester by definition: 196 d = 28+0 wk, 300 d = 42+6 wk.
    "ga": [196, 300],          # gestational age at scan, days
    # 28 wk ~71 mm, 40 wk ~94 mm; bounds allow ~+/-3 SD either side.
    "bpd": [50, 115],          # biparietal diameter, mm
    # 28 wk ~26 cm, 40 wk ~35 cm. An hc above 40 cm is a keying error, not a head.
    "hc": [20, 40],            # head circumference, cm
    # 28 wk ~23 cm, 42 wk ~37 cm; 18 cm admits severe growth restriction.
    "ac": [18, 42],            # abdominal circumference, cm
    # 28 wk ~5.3 cm, 42 wk ~7.9 cm.
    "fl": [4.0, 9.0],          # femur length, cm
    "m_age": [13, 55],         # maternal age, years
    # Cerebroplacental ratio = MCA-PI / UA-PI.
    "cpr": [0.2, 3.0],         # cerebroplacental ratio
    # Middle cerebral artery peak systolic velocity. Zero is not a measurement.
    "psv": [10, 130],          # peak systolic velocity, cm/s
    # 28 wk severe restriction ~600 g; upper bound admits macrosomia >4.5 kg.
    "efw": [500, 5500],        # estimated fetal weight, g
    "m_height": [120, 200],    # maternal height, cm
    "m_weight": [35, 200],     # maternal weight, kg
    # Amniotic fluid index: oligohydramnios <5, polyhydramnios >24.
    "afi": [0.0, 40.0],        # amniotic fluid index, cm
    # A resistance index is a ratio bounded by 0 and 1 by construction.
    "ute_ari": [0.0, 1.0],     # uterine artery resistance index
    # Third-trimester normal ~0.5-1.2; severe pre-eclampsia can exceed 2.5.
    "ute_api": [0.0, 3.0],     # uterine artery pulsatility index
    # Raised with absent or reversed end-diastolic flow.
    "umb_api": [0.0, 3.0],     # umbilical artery pulsatility index
}

#: Features stored in millimetres at screening time but in centimetres in the final
#: tables, so the screening bound is the tabled bound x 10.
_MM_AT_SCREENING = ("hc", "ac", "fl")


def screening_bounds(feature, in_millimetres=False):
    """Plausible (lower, upper) bound for ``feature`` used by cohort screening."""
    low, high = CONTINUOUS_FEATURE_LOGICAL_RANGE[feature]
    if in_millimetres and feature in _MM_AT_SCREENING:
        return low * 10, high * 10
    return low, high


def remove_illogical_values(df, keep_nan=True, return_indices=False):
    """Drop out-of-range rows in place."""
    removed = set()
    for col, (lower, upper) in CONTINUOUS_FEATURE_LOGICAL_RANGE.items():
        if col not in df.columns:
            continue
        in_range = (df[col] >= lower) & (df[col] <= upper)
        if keep_nan:
            in_range |= df[col].isna()
        removed.update(df.index[~in_range])

    df.drop(index=removed, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return removed if return_indices else None


def remove_duplicates(df, drop_first_column=True, save_path=None, drop_cols=None):
    """Drop exact duplicate records, optionally logging them to ``save_path``."""
    if drop_first_column:
        df = df.iloc[:, 1:]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.reset_index(drop=True)
    df.columns = df.columns.str.strip()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    if save_path is not None:
        df[df.duplicated(keep=False)].to_csv(save_path, index=False)

    return df.drop_duplicates(keep="first").reset_index(drop=True)


def _writeable_float_matrix(df):
    """A contiguous, writeable float array of ``df``.

    scikit-learn's iterative imputer writes into the array it is handed, and a
    frame assembled by concatenation or slicing can expose a read-only buffer -
    which surfaces as "assignment destination is read-only" from deep inside the
    imputer. Handing it an array also keeps fit and transform on the same
    representation, so no feature-name warning is raised.
    """
    return np.array(df.to_numpy(dtype=float), copy=True, order="C")


def fit_iterative_imputer(df, random_state=SEED, verbose=False):
    """Fit an iterative (MICE-style) imputer on ``df``.

    Fitting and applying are separate so the imputer can be fitted on ONE fold's
    training partition and then applied unchanged to that fold's held-out rows,
    which is what the Methods require: "remaining within-feature missing values
    were completed by iterative imputation fitted on the training partition".
    Fitting once on the pooled development block instead lets a held-out record
    shape the model that fills its own fold's training rows.

    ``keep_empty_features`` is set so a column that is entirely missing in this
    fold's training rows is filled with a constant rather than silently dropped,
    which would change the feature space from one fold to the next.
    """
    if verbose:
        total_missing = int(df.isna().sum().sum())
        print("Missing values per column:\n", df.isna().sum())
        print(f"Total missing values: {total_missing}")
        print(f"Overall missing data percentage: {total_missing / df.size * 100:.2f}%")

    try:
        imputer = IterativeImputer(random_state=random_state, keep_empty_features=True)
    except TypeError:  # scikit-learn < 1.2
        imputer = IterativeImputer(random_state=random_state)
    imputer.fit(_writeable_float_matrix(df))
    return imputer


def apply_iterative_imputer(imputer, df):
    """Apply a fitted imputer, filling ONLY the missing cells.

    Observed values are written back unchanged rather than taken from the
    transform output, so a fold's real measurements can never be perturbed by the
    imputation model and the column order and index survive intact.
    """
    filled = pd.DataFrame(
        imputer.transform(_writeable_float_matrix(df)),
        columns=df.columns,
        index=df.index,
    )
    return df.where(df.notna(), filled)


def iterative_impute(df, random_state=SEED, verbose=True):
    """Fit an iterative imputer on ``df`` and complete it in one step.

    Convenience wrapper for the whole-frame case. Anywhere a train/held-out split
    exists, use :func:`fit_iterative_imputer` on the training rows and
    :func:`apply_iterative_imputer` on both sides instead.
    """
    imputer = fit_iterative_imputer(df, random_state=random_state, verbose=verbose)
    return apply_iterative_imputer(imputer, df)
