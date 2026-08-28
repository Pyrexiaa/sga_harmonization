"""India cohort: two data drops -> one analysis-ready third-trimester frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sga.config import CHART, PROCESSED_DATA_DIR
from sga.data.centiles import compute_efw, merge_groundtruth
from sga.data.cleaning import remove_duplicates, remove_illogical_values, screening_bounds
from sga.data.encoding import is_numeric
from sga.data.screening import ScreeningLog

#: Default locations of the two raw exports under ``RAW_DATA_DIR``.
RAW_MAY8_SUBPATH = Path("IndiaDatasets") / "india_may8_cleaned_latest.csv"
RAW_APRIL8_SUBPATH = Path("IndiaDatasets") / "Data_April_8_to_share.csv"

#: Verbose survey headers in the April-8 export -> harmonized names.
APRIL8_RENAME = {
    "maternalagecompletedyears": "m_age",
    "heightinmeters": "m_height",
    "weightinkilograms": "m_weight",
    "lastpregnancysga": "last_preg_sga",
    "lastpregnancyfgr": "last_preg_fgr",
    "lastpregnancynormalbaby": "last_preg_normal",
    "pregnancyinducedhypertension": "hypertension_0",
    "essentialhypertension": "hypertension_1",
    "gestationaldm": "diabetes_0",
    "pregestationaldm": "diabetes_1",
    "gaatassesmentincompletedweek": "ga",
    "efwgrams": "efw",
    "efwcentile": "efw_centile",
    "meanutapi": "ute_api",
    "umbilicalapi": "umb_api",
    "fetalpresentation": "presentation",
    "Placentalthikness": "placenta_thickness",
    "singleverticalpocket": "single_vertical_pocket",
    "knownhighriskofpe": "high_risk_pe",
    "knownhighriskoffgr": "high_risk_fgr",
    "gaatdeliveryweeks": "birth_ga",
    "birthweightgrams": "bw",
}

#: Columns kept from the April-8 export.
APRIL8_COLUMNS = [
    "m_age", "m_height", "m_weight", "last_preg_sga", "last_preg_fgr",
    "last_preg_normal", "smoking", "hypertension_0", "hypertension_1",
    "diabetes_0", "diabetes_1", "ga", "bpd", "hc", "ac", "fl", "ute_api",
    "umb_api", "cpr", "presentation", "placenta_thickness",
    "single_vertical_pocket", "high_risk_pe", "high_risk_fgr", "bw", "gender",
    "prev_failed_preg", "birth_ga", "sga",
]

#: Columns kept from the May-8 export (``utapi_mean`` is renamed afterwards).
MAY8_COLUMNS = [
    "m_age", "m_height", "m_weight", "last_preg_sga", "last_preg_fgr",
    "last_preg_normal", "smoking", "hypertension_0", "hypertension_1",
    "diabetes_0", "diabetes_1", "ga", "bpd", "hc", "ac", "fl", "utapi_mean",
    "umb_api", "cpr", "presentation", "placenta_thickness",
    "single_vertical_pocket", "high_risk_pe", "high_risk_fgr", "bw", "gender",
    "prev_failed_preg", "birth_ga", "sga",
]

#: Free-text yes/no history fields recoded to 0/1.
YES_NO_COLUMNS = [
    "last_preg_sga", "last_preg_fgr", "last_preg_normal", "smoking",
    "hypertension_0", "hypertension_1", "diabetes_0", "diabetes_1",
    "high_risk_pe",
]

#: Binary/count fields stored as integers in the released tables.
INTEGER_COLUMNS = YES_NO_COLUMNS + ["prev_failed_preg"]

#: Postnatal fields that must not be visible to an antenatal model.
POSTNATAL_COLUMNS = ["birth_ga", "hc_actual", "bl", "bw"]

#: Third trimester starts at 28 completed weeks; ``ga`` is recorded in days.
THIRD_TRIMESTER_DAY = 28 * 7


def _apply_biometry_filters(df, log=None):
    """Drop scans whose biometry is outside the plausible third-trimester range."""

    def keep(column, in_millimetres):
        low, high = screening_bounds(column, in_millimetres=in_millimetres)
        return ((df[column] >= low) & (df[column] <= high)) | df[column].isna()

    # bpd is millimetres in both the raw export and the final tables; ac and fl are
    # still millimetres here and are converted downstream.
    for column, in_mm in (("bpd", False), ("ac", True), ("fl", True), ("cpr", False)):
        if column in df.columns:
            df = df[keep(column, in_mm)]

    df = df.copy()
    df["hc"] = df["hc"] / 10  # mm -> cm
    hc_low, hc_high = screening_bounds("hc")
    df = df[(df["hc"] >= hc_low) & (df["hc"] <= hc_high) | df["hc"].isna()]
    if log is not None:
        log.record("implausible fetal biometry", len(df))
    return df


def _extract_count(value):
    """Pull the leading integer out of a free-text gravidity entry."""
    if pd.isna(value):
        return np.nan
    digits = "".join(filter(str.isdigit, str(value)))
    return int(digits) if digits else np.nan


def _parity(decision, gravida):
    """Convert the ``Nulliparous``/``Parous`` decision into a parity count."""
    if decision == "Nulliparous":
        return 0
    if decision == "Parous":
        return max(gravida - 1, 0)
    return np.nan


def preprocess_india_april8(raw_csv, chart=CHART, log=None, drop_columns=None):
    """Clean the April-8 Indian export."""
    mapping_dir = Path(PROCESSED_DATA_DIR / "IndiaDataset")
    mapping_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_csv)
    log = log or ScreeningLog("India (April-8 drop)", len(raw))

    df = remove_duplicates(
        raw, drop_first_column=True, save_path=mapping_dir / "april8_duplicates.csv"
    )
    log.record("duplicate records", len(df))

    # Previous failed pregnancies = gravidity - parity.
    gravida = df["gravida"].apply(_extract_count)
    parity = pd.Series(
        [_parity(decision, count) for decision, count in zip(df["para"], gravida)],
        index=df.index,
    )
    df["prev_failed_preg"] = gravida - parity

    df = df.rename(columns=APRIL8_RENAME)
    df = df.dropna(subset=["gender"])
    log.record("fetal sex not recorded", len(df))

    df = merge_groundtruth(df, others=False, divide_by_1000=True, chart=chart)
    df = df[df["gender"].isin(["M", "F"])]
    df["gender"] = df["gender"].map({"M": 1, "F": 0}).astype(int)
    log.record("unusable fetal-sex code", len(df))

    df = df[APRIL8_COLUMNS].copy()
    # This export records gestational age in completed weeks.
    df["ga"] = df["ga"] * 7
    df["birth_ga"] = df["birth_ga"] * 7

    df["m_weight"] = pd.to_numeric(df["m_weight"], errors="coerce")
    for column in YES_NO_COLUMNS:
        df[column] = pd.to_numeric(
            df[column].str.lower().replace({"yes": 1, "no": 0}), errors="coerce"
        )

    df = df.dropna(axis=1, how="all")
    df = df.drop(columns=list(drop_columns or []), errors="ignore")
    return _apply_biometry_filters(df, log), log


def preprocess_india_may8(raw_csv, chart=CHART, log=None, drop_columns=None):
    """Clean the May-8 Indian export."""
    mapping_dir = Path(PROCESSED_DATA_DIR / "IndiaDataset")
    mapping_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_csv)
    log = log or ScreeningLog("India (May-8 drop)", len(raw))

    df = remove_duplicates(
        raw, drop_first_column=True, save_path=mapping_dir / "may8_duplicates.csv"
    )
    log.record("duplicate records", len(df))

    # The shipped labels are recomputed from the chosen reference chart.
    df = df.drop(columns=["lbw", "sga", "sc", "cur_sga"], errors="ignore")
    df["gender"] = df["gender"].map({1: "M", 0: "F"})
    df = merge_groundtruth(df, others=False, divide_by_1000=True, chart=chart)
    df["gender"] = df["gender"].map({"M": 1, "F": 0})

    df = df[MAY8_COLUMNS].rename(columns={"utapi_mean": "ute_api"})
    df = df.drop(columns=list(drop_columns or []), errors="ignore")

    # Entries such as ">99" cannot be interpreted and disqualify the record.
    df = df[df.map(is_numeric).all(axis=1)]
    log.record("non-numeric measurement entries", len(df))

    df = df.copy()
    df["sga"] = df["sga"].astype(int)
    return _apply_biometry_filters(df, log), log


def manual_remove_rare_cases(df):
    """Neutralise implausible values and enforce the third-trimester window."""
    bounds = {
        "bpd": df.get("bpd", pd.Series(dtype=float)) < 50,
        "psv": df.get("psv", pd.Series(dtype=float)) < 10,
        "ute_api": df.get("ute_api", pd.Series(dtype=float)) > 2,
        "prev_failed_preg": df.get("prev_failed_preg", pd.Series(dtype=float)) < 0,
    }
    df = df.copy()
    for column, mask in bounds.items():
        if column in df.columns:
            df.loc[mask, column] = np.nan
    return df[df["ga"] >= THIRD_TRIMESTER_DAY]


def combine_india_drops(april8, may8, log=None):
    """Reduce both drops to their shared columns and concatenate them."""
    shared = may8.columns.intersection(april8.columns).difference(["Unnamed: 0"])
    df = pd.concat([may8[shared], april8[shared]], axis=0).reset_index(drop=True)

    empty = list(df.columns[df.isna().all()]) + (
        ["efw_centile"] if "efw_centile" in df.columns else []
    )
    if empty:
        print(f"  Columns dropped because they carry no data: {empty}")
    df = df.drop(columns=empty, errors="ignore")

    log = log or ScreeningLog("India (combined)", len(df))
    df = manual_remove_rare_cases(df)
    log.record("no third-trimester scan after 28 weeks", len(df))

    df = df.copy()
    df[["ac", "fl"]] = df[["ac", "fl"]] / 10  # mm -> cm
    df.loc[df["bw"] >= 10000, "bw"] /= 10     # mis-keyed decimal place
    df["efw"] = df.apply(compute_efw, axis=1)

    # The April-8 export records maternal height in METRES ("heightinmeters").
    if "m_height" in df.columns:
        height = pd.to_numeric(df["m_height"], errors="coerce")
        if height.notna().any() and height.median() < 3:
            df["m_height"] = height * 100
            print("  Converted m_height from metres to centimetres.")
        else:
            df["m_height"] = height

    # Apply the shared plausibility ranges here, as the Malaysian preprocessor already
    # does, so both cohorts leave preprocessing inside the same bounds and nothing is
    # dropped later during fold construction.
    before = len(df)
    remove_illogical_values(df, keep_nan=True)
    if before != len(df):
        print(f"  Dropped {before - len(df)} record(s) outside the plausible ranges.")
    log.record("implausible measurements", len(df))

    print(f"India tri-3 records: {len(df)}; SGA: {int((df['sga'] == 1).sum())}")
    return df, log


def cast_integer_columns(df):
    """Force the binary/count history fields to non-negative integers."""
    for column in INTEGER_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype(int).clip(lower=0)
    return df


def drop_postnatal_columns(df, extra=None):
    """Remove postnatal fields that must not be visible to an antenatal model."""
    return df.drop(columns=POSTNATAL_COLUMNS + list(extra or []), errors="ignore")
