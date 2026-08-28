"""Birthweight / EFW centile lookup and SGA labelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sga.config import CENTILE_DIR

CENTILE_PERCENTILES = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
_CENTILE_COLUMNS = ["p_3", "p_5", "p_10", "p_50", "p_90", "p_95", "p_97"]

# chart -> (reference file, divisor when birthweight is expressed in grams)
_CHARTS = {
    "i21": ("I21_BW.csv", 1000),
    "hadlock": ("HL_EFW.csv", 1),
    "who": ("WHO_EFW.csv", 1),
    "malaysia": ("MSIA_EFW.csv", 1),
}


def _load_efw_chart(filename: str) -> pd.DataFrame:
    """Load a gestational-week EFW chart and interpolate it to daily resolution."""
    chart = pd.read_csv(CENTILE_DIR / filename)
    chart.columns = chart.columns.str.lower()
    chart = chart.rename(columns={"ga": "birth_ga_weeks"})
    chart["birth_ga"] = (chart["birth_ga_weeks"] * 7).astype(int)

    all_days = pd.DataFrame(
        {"birth_ga": range(chart["birth_ga"].min(), chart["birth_ga"].max() + 1)}
    )
    chart = all_days.merge(chart, on="birth_ga", how="left")
    chart = chart.interpolate(method="linear").drop(columns="birth_ga_weeks")
    return chart


def merge_groundtruth(df, others=True, divide_by_1000=True, chart="i21"):
    """Attach reference centiles to ``df`` and derive the binary SGA label."""
    if chart not in _CHARTS:
        raise ValueError(f"Unsupported chart {chart!r}; choose from {sorted(_CHARTS)}")

    filename, gram_divisor = _CHARTS[chart]
    if chart == "i21":
        divisor = gram_divisor if divide_by_1000 else 1
        reference = pd.read_csv(CENTILE_DIR / filename)
        reference.columns = reference.columns.str.lower()
        reference = reference.rename(columns={"ga": "birth_ga"})
        df = df.merge(reference, on=["birth_ga", "gender"], how="left")
    else:
        divisor = gram_divisor if divide_by_1000 else 0.001
        df = df.merge(_load_efw_chart(filename), on="birth_ga", how="left")

    # `sga` first:
    df["sga"] = (df["bw"] / divisor <= df["p_10"]).astype(int)
    if others:
        df["lbw"] = (df["bw"] / divisor <= 2.5).astype(int)
        df["sc"] = (df["cur_sga"] ^ df["sga"]).astype(int)

    return df.drop(columns=df.columns.intersection(_CENTILE_COLUMNS))


def compute_efw(row):
    """Hadlock estimated fetal weight from HC, AC and FL (grams)."""
    if pd.isna(row["ac"]) or pd.isna(row["hc"]) or pd.isna(row["fl"]):
        return np.nan
    log_efw = (
        1.326
        + 0.0107 * row["hc"]
        + 0.0438 * row["ac"]
        + 0.158 * row["fl"]
        - 0.00326 * row["ac"] * row["fl"]
    )
    return round(10**log_efw)


def compute_efw_centile(df):
    """Interpolate each scan's EFW onto the reference centile curve."""
    reference = pd.read_excel(CENTILE_DIR / "EFW_Centile.xlsx").rename(columns={"GA": "ga"})
    df = df.merge(reference, on="ga", how="left")

    percentiles = list(reference.columns)[1:]
    bounds = df[CENTILE_PERCENTILES].to_numpy()
    efw = df["efw"].to_numpy()

    centiles = []
    for value, row_bounds in zip(efw, bounds):
        if not (row_bounds[0] <= value <= row_bounds[-1]):
            centiles.append(0)
            continue
        for j, upper in enumerate(row_bounds):
            if value == upper:
                centiles.append(percentiles[j])
                break
            if value < upper:
                span = upper - row_bounds[j - 1]
                centiles.append(
                    percentiles[j]
                    - (percentiles[j] - percentiles[j - 1]) * (upper - value) / span
                )
                break

    df["efw_centile"] = centiles
    df = df[df["efw_centile"] != 0].copy()
    df["cur_sga"] = (df["efw_centile"] <= 10).astype(int)
    return df.drop(columns=CENTILE_PERCENTILES)


def assign_sga_label(row):
    """Four-way agreement code between the birthweight label and EFW centile."""
    small_on_scan = row["efw_centile"] < 10
    if row["sga"] == 0:
        return 1 if small_on_scan else 0
    return 2 if small_on_scan else 3
