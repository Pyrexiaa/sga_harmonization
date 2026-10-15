"""Malaysia cohort: raw hospital export -> analysis-ready third-trimester frame."""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sga.config import CHART, PROCESSED_DATA_DIR
from sga.data.centiles import compute_efw, merge_groundtruth
from sga.data.cleaning import remove_duplicates, remove_illogical_values
from sga.data.screening import ScreeningLog

#: Default location of the raw Malaysian export under ``RAW_DATA_DIR``.
RAW_SUBPATH = Path("MalaysiaDatasets") / "cleaned_batch_3.csv"

#: Raw export column names -> the harmonized names used across the project.
RENAME_COLUMNS = {
    "baby_gender": "gender",
    "birth_weight": "bw",
    "birth_length": "bl",
    "head_circumference": "hc_actual",
    "WHO_cur_sga": "cur_sga",
    "WHO_sga": "sga",
    "WHO_sc": "sc",
    "uari": "ute_ari",
    "uapi": "ute_api",
}

#: Columns retained from the raw export, in the order used downstream.
SELECTED_COLUMNS = [
    "id", "ga", "bpd", "hc", "ac", "fl", "afi", "uari", "uapi", "m_age",
    "placenta_site", "af", "gender", "nf", "cpr", "humerus", "tcd", "psv",
    "mode_of_delivery", "admission_place", "apgar_score_1min",
    "apgar_score_5min", "birth_ga", "bw", "bl",
]

CATEGORICAL_COLUMNS = [
    "placenta_site", "af", "gender", "mode_of_delivery", "admission_place",
]

#: Measurements available for too few pregnancies to be usable.
SPARSE_COLUMNS = ["nf", "humerus", "tcd"]

#: Canonical spellings the free-text ``mode_of_delivery`` field is snapped to.
DELIVERY_MODE_TERMS = [
    "SVD", "ELECTIVE CAESAREAN", "EMERGENCY CAESAREAN", "VACUUM", "CAESAREAN",
    "FORCEP", "BREECH DELIVERY", "BORN AT HOME", "CAESAREAN - HYSTEROTOMY",
    "BBA @ WARD", "BBA OTHERS", "BBA @ EMERGENCY",
]

#: Clinical grouping of the delivery modes:
DELIVERY_MODE_GROUPS = {
    "SVD": "SVD",
    "CAESAREAN": "CAESAREAN",
    "CAESAREAN - HYSTEROTOMY": "CAESAREAN",
    "ELECTIVE CAESAREAN": "CAESAREAN",
    "EMERGENCY CAESAREAN": "CAESAREAN",
    "VACUUM": "ASSISTED PREGNANCY",
    "FORCEP": "ASSISTED PREGNANCY",
    "BREECH DELIVERY": "ASSISTED PREGNANCY",
    "BBA @ EMERGENCY": "BIRTH OUTSIDE OF HOSPITAL",
    "BBA @ WARD": "BIRTH OUTSIDE OF HOSPITAL",
    "BBA OTHERS": "BIRTH OUTSIDE OF HOSPITAL",
    "BORN AT HOME": "BIRTH OUTSIDE OF HOSPITAL",
}

PLACENTA_SITE_GROUPS = {
    "anterior": "Anterior Placenta",
    "anterior high": "Anterior Placenta",
    "anterior left": "Anterior Placenta",
    "anterior low": "Anterior Placenta",
    "anterior right": "Anterior Placenta",
    "Right lateral anterior high": "Anterior Placenta",
    "posterior high": "Posterior Placenta",
    "posterior left": "Posterior Placenta",
    "posterior low": "Posterior Placenta",
    "posterior right": "Posterior Placenta",
    "right lateral posterior": "Posterior Placenta",
    "lateral left": "Lateral Placenta",
    "lateral low lying": "Lateral Placenta",
    "lateral right": "Lateral Placenta",
    "lateral right low lying": "Lateral Placenta",
    "Lateral left upper segment": "Lateral Placenta",
    "Lateral left-posterior lowlying": "Lateral Placenta",
    "Lateral right upper segment": "Lateral Placenta",
    "fundal": "Fundal Placenta",
    "Placenta praevia": "Placenta Previa",
    "placenta praevia": "Placenta Previa",
    "placenta praevia major": "Placenta Previa",
    "placenta praevia type 1": "Placenta Previa",
    "placenta praevia type 2": "Placenta Previa",
    "placenta praevia type 3": "Placenta Previa",
}

ADMISSION_PLACE_GROUPS = {
    "NURSERY": "Inpatient Wards",
    "PAEDS WARD": "Inpatient Wards",
    "POSTNATAL WARD": "Inpatient Wards",
    "PICU": "Specialized Care",
    "SCN": "Specialized Care",
    "OTHERS": "Others",
}

#: Third trimester starts at 28 completed weeks; ``ga`` is recorded in days.
THIRD_TRIMESTER_DAY = 28 * 7

#: Postnatal fields that must not leak into a prediction made before delivery.
POSTNATAL_COLUMNS = [
    "hc_actual", "bl", "bw", "apgar_score_1min", "apgar_score_5min",
    "mode_of_delivery", "admission_place",
]


def correct_typo(value, choices=DELIVERY_MODE_TERMS, cutoff=0.9):
    """Snap a free-text entry to its closest canonical spelling."""
    if not isinstance(value, str):
        return value
    match = get_close_matches(value, choices, n=1, cutoff=cutoff)
    return match[0] if match else value


def encode_label(df, column, mapping_path=None):
    """Integer-encode a categorical column and persist its code book."""
    encoded = df.copy()
    encoder = LabelEncoder()
    encoded[column] = encoder.fit_transform(encoded[column].fillna("Unknown"))

    mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    if mapping_path is not None:
        mapping_path = Path(mapping_path)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(mapping.items(), columns=["Original", "Encoded"]).to_csv(
            mapping_path, index=False
        )

    unknown = mapping.get("Unknown")
    if unknown is not None:
        encoded[column] = encoded[column].replace(unknown, np.nan)
    return encoded


def _regroup(df, column, groups):
    """Collapse a free-text column onto its clinical grouping, reporting misses."""
    unmapped = set(df[column].dropna().unique()) - set(groups)
    if unmapped:
        print(f"  [{column}] unmapped labels ignored: {sorted(unmapped)}")
    df[column] = df[column].map(groups)
    return df


def _coerce_numeric(df, skip):
    """Replace non-numeric sentinel strings (e.g. ``">99"``) with ``NaN``."""
    offenders = set()
    for column in df.columns:
        if column in skip:
            continue
        try:
            df[column] = pd.to_numeric(df[column], errors="raise", downcast="float")
        except (ValueError, TypeError):
            values = df[column].unique()
            offenders.update(
                value
                for value in values
                if isinstance(value, str) and not value.replace(".", "", 1).isdigit()
            )
    if offenders:
        print(f"  Non-numeric entries replaced with NaN: {sorted(offenders)}")
        df = df.replace(list(offenders), np.nan)
    return df


def manual_remove_rare_cases(df):
    """Drop the handful of scans that are implausible on every measurement."""
    bounds = {"bpd": 2.4, "ga": 300, "psv": 10, "ute_api": 1.4, "ute_ari": 0.9}
    present = [col for col in bounds if col in df.columns]
    if not present:
        return df

    implausible = df[present[0]] < bounds[present[0]]
    for column in present[1:]:
        implausible &= df[column] < bounds[column]
    return df[~implausible].reset_index(drop=True)


def preprocess_malaysia(
    raw_csv,
    chart=CHART,
    log=None,
    mapping_dir=None,
    drop_columns=None,
):
    """Clean the raw Malaysian export into an analysis-ready tri-3 frame."""
    mapping_dir = Path(mapping_dir or PROCESSED_DATA_DIR / "MalaysiaDataset")
    mapping_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_csv)
    log = log or ScreeningLog("Malaysia", len(raw))

    df = remove_duplicates(
        raw, drop_first_column=True, save_path=mapping_dir / "duplicates.csv"
    )
    log.record("duplicate records", len(df))

    df = df.rename(columns=RENAME_COLUMNS)
    df = df[[RENAME_COLUMNS.get(c, c) for c in SELECTED_COLUMNS]]

    # INTERGROWTH-21st needs the reference chart's M/F coding; the numeric coding is
    # restored immediately afterwards.
    df["gender"] = df["gender"].map({1: "M", 0: "F"})
    df = merge_groundtruth(df, others=False, divide_by_1000=False, chart=chart)
    df["gender"] = df["gender"].map({"M": 1, "F": 0})
    df["sga"] = df["sga"].astype(int)

    df = df[(df["mode_of_delivery"] != "NONE") & (df["admission_place"] != "NONE")]
    log.record("no recorded delivery mode or admission place", len(df))

    df["mode_of_delivery"] = df["mode_of_delivery"].apply(correct_typo)
    df = _regroup(df, "mode_of_delivery", DELIVERY_MODE_GROUPS)
    df = _regroup(df, "placenta_site", PLACENTA_SITE_GROUPS)
    df = _regroup(df, "admission_place", ADMISSION_PLACE_GROUPS)

    df = df.dropna(subset=CATEGORICAL_COLUMNS)
    log.record("incomplete categorical antenatal/postnatal fields", len(df))

    for column in CATEGORICAL_COLUMNS:
        df = encode_label(df, column, mapping_dir / f"{column}_mapping_{chart}.csv")

    df = df[df["ga"] >= THIRD_TRIMESTER_DAY]
    log.record("no third-trimester scan after 28 weeks", len(df))

    for column in drop_columns or []:
        df = df.drop(columns=column, errors="ignore")

    # Birthweight is keyed either in kilograms or in grams; anything in between is a
    # data-entry error.
    df = df[(df["bw"] != 0) & ((df["bw"] < 10) | (df["bw"] >= 1000))]
    df.loc[df["bw"] >= 1000, "bw"] /= 1000
    log.record("implausible birthweight", len(df))

    df = df[(df["bpd"] >= 50) | df["bpd"].isna()]
    df = df[(df["m_age"] >= 16) | df["m_age"].isna()]
    df = df[(df["psv"] > 0) | df["psv"].isna()]
    log.record("implausible bpd / maternal age / psv", len(df))

    df = df.copy()
    df[["hc", "ac", "fl"]] = df[["hc", "ac", "fl"]] / 10  # mm -> cm
    df["efw"] = df.apply(compute_efw, axis=1)

    df = _coerce_numeric(df, skip=CATEGORICAL_COLUMNS)
    df = df.drop(columns=SPARSE_COLUMNS, errors="ignore")

    remove_illogical_values(df, keep_nan=True)
    log.record("measurement outside physiological range", len(df))

    print(f"Malaysia tri-3 records: {len(df)}; SGA: {int((df['sga'] == 1).sum())}")
    return df, log


def drop_postnatal_columns(df, extra=None):
    """Remove postnatal fields that must not be visible to an antenatal model."""
    return df.drop(columns=POSTNATAL_COLUMNS + list(extra or []), errors="ignore")
