"""Birthweight / EFW centile lookup and SGA labelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sga.config import CENTILE_DIR, EFW_CENTILE_RULE

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


def load_efw_centile_reference(path=None):
    """EFW reference centiles by gestational age in days (``RefCentile/EFW_Centile.xlsx``)."""
    reference = pd.read_excel(path or CENTILE_DIR / "EFW_Centile.xlsx").rename(columns={"GA": "ga"})
    reference["ga"] = reference["ga"].astype(int)
    return reference


def efw_centile(df, reference=None, rule=EFW_CENTILE_RULE):
    """EFW centile of every scan, keeping every row.

    Inside the tabulated 2.5th-97.5th range the centile is linearly interpolated;
    beyond it, it is linearly extrapolated from the two outermost centiles and
    bounded to 0.01-99.99, so the most extreme (and most SGA-relevant) scans are
    not dropped from the comparison. The < 10th centile rule is unaffected by the
    extrapolation, because any EFW below the tabulated 2.5th centile is below the
    10th. Returns ``efw_centile``, ``centile_status`` and ``below_rule`` aligned to
    ``df.index``.
    """
    reference = load_efw_centile_reference() if reference is None else reference
    levels = np.array([float(c) for c in reference.columns[1:]])
    lookup = {
        int(row[0]): np.asarray(row[1:], dtype=float)
        for row in reference.itertuples(index=False)
    }

    ga = pd.to_numeric(df["ga"], errors="coerce").round().to_numpy()
    efw = pd.to_numeric(df["efw"], errors="coerce").to_numpy(dtype=float)
    centile = np.full(len(df), np.nan)
    status = np.full(len(df), "in_range", dtype=object)

    for i, (day, weight) in enumerate(zip(ga, efw)):
        curve = lookup.get(int(day)) if np.isfinite(day) else None
        if curve is None or not np.isfinite(weight):
            status[i] = "no_reference"
        elif weight < curve[0]:
            slope = (levels[1] - levels[0]) / (curve[1] - curve[0])
            centile[i] = max(0.01, levels[0] + slope * (weight - curve[0]))
            status[i] = "below_table"
        elif weight > curve[-1]:
            slope = (levels[-1] - levels[-2]) / (curve[-1] - curve[-2])
            centile[i] = min(99.99, levels[-1] + slope * (weight - curve[-1]))
            status[i] = "above_table"
        else:
            centile[i] = float(np.interp(weight, curve, levels))

    out = pd.DataFrame({"efw_centile": centile, "centile_status": status}, index=df.index)
    out["below_rule"] = (out["efw_centile"] < rule).astype(int)
    return out
