"""Cohort characteristics of the prepared tables (manuscript Table 1).

Usage:
    python -m scripts.01c_cohort_characteristics --cohort both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sga import config
from sga.config import set_seed
from sga.data.statistics import (
    get_count_categorical_feature,
    get_mean_continuous_feature,
    normality_and_parametric_test,
    save_results_to_csv,
    save_results_to_excel,
)

#: Columns that are identifiers or pipeline bookkeeping, never characteristics.
NON_FEATURE_COLUMNS = {"Unnamed: 0", "id", "fold", config.LABEL}

CATEGORICAL_BY_COHORT = {
    config.MALAYSIA_SUBDIR: sorted(
        set(config.MALAYSIA_CATEGORICAL) | set(config.MALAYSIA_MULTICLASS_FEATURES)
    ),
    config.INDIA_SUBDIR: sorted(set(config.INDIA_CATEGORICAL)),
}


def load_cohort_table(subdir, chart, base_dir=None):
    """Pool the complete-case and add-on tables of one cohort."""
    base = Path(base_dir or config.TRAINING_DATA_DIR) / subdir
    complete = pd.read_csv(base / f"tri3_{chart}.csv")
    add_on = pd.read_csv(base / f"tri3_add_on_{chart}.csv")
    return pd.concat([complete, add_on], axis=0).reset_index(drop=True)


def describe_cohort(df, subdir, save_dir, chart):
    """Compute and persist the Table 1 statistics for one cohort."""
    positive = df[df[config.LABEL] == 1]
    negative = df[df[config.LABEL] == 0]
    print(
        f"\n{subdir}: {len(df)} records, {len(positive)} SGA "
        f"({len(positive) / max(len(df), 1) * 100:.2f}%), {len(negative)} AGA"
    )

    categorical = [c for c in CATEGORICAL_BY_COHORT[subdir] if c in df.columns]
    continuous = [
        c
        for c in df.columns
        if c not in categorical and c not in NON_FEATURE_COLUMNS
    ]

    categorical_results, continuous_results = [], []
    for feature in categorical:
        get_count_categorical_feature(
            positive, negative, feature, results=categorical_results
        )
        normality_and_parametric_test(
            df, positive, negative, feature,
            continuous=False, results=categorical_results,
        )
    for feature in continuous:
        get_mean_continuous_feature(
            positive, negative, feature, results=continuous_results
        )
        normality_and_parametric_test(
            df, positive, negative, feature,
            continuous=True, results=continuous_results,
        )

    for name, results in (
        ("categorical", categorical_results),
        ("continuous", continuous_results),
    ):
        stem = save_dir / f"table1_{subdir.lower()}_{name}_{chart}"
        save_results_to_csv(results, f"{stem}.csv")
        save_results_to_excel(results, f"{stem}.xlsx")
        print(f"  wrote {stem}.csv / .xlsx")

    return categorical_results, continuous_results


def write_feature_availability(save_dir, chart, data_dir=None):
    """Write Appendix Table S1: which features each cohort actually measures."""
    rows = []
    for subdir in (config.MALAYSIA_SUBDIR, config.INDIA_SUBDIR):
        try:
            df = load_cohort_table(subdir, chart, data_dir)
        except FileNotFoundError:
            print(f"  {subdir}: prepared table missing; skipped in Table S1.")
            continue
        for feature in config.COMMON_FEATURES + config.ALL_CROSS_DOMAIN_FEATURES:
            present = feature in df.columns
            rows.append({
                "cohort": subdir,
                "feature": feature,
                "group": (
                    "common" if feature in config.COMMON_FEATURES else "cohort-specific"
                ),
                "measured": "Yes" if present else "No",
                "n_observed": int(df[feature].notna().sum()) if present else 0,
                "pct_observed": (
                    round(100 * df[feature].notna().mean(), 2) if present else 0.0
                ),
            })

    if not rows:
        return None

    table = pd.DataFrame(rows)
    stem = save_dir / f"tableS1_feature_availability_{chart}"
    table.to_csv(f"{stem}.csv", index=False)
    table.to_excel(f"{stem}.xlsx", index=False)
    print(f"  wrote {stem}.csv / .xlsx")

    for cohort, group in table[table["measured"] == "Yes"].groupby("cohort"):
        print(f"  {cohort}: {len(group)} features measured")
    return table


def main():
    """Produce manuscript Table 1 and Appendix Table S1 for one or both cohorts."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--cohort", choices=("malaysia", "india", "both"), default="both")
    add("--data-dir", type=Path, default=None)
    add("--output-dir", type=Path, default=None)
    add("--chart", default=config.CHART)
    args = parser.parse_args()
    set_seed(config.SEED)

    save_dir = args.output_dir or config.results_path("tables")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    wanted = (
        [config.MALAYSIA_SUBDIR, config.INDIA_SUBDIR]
        if args.cohort == "both"
        else [config.MALAYSIA_SUBDIR if args.cohort == "malaysia" else config.INDIA_SUBDIR]
    )
    for subdir in wanted:
        df = load_cohort_table(subdir, args.chart, args.data_dir)
        describe_cohort(df, subdir, save_dir, args.chart)

    print("\nAppendix Table S1 -- feature availability per cohort")
    write_feature_availability(save_dir, args.chart, args.data_dir)

    print(f"\nTable 1 statistics and Table S1 written to {save_dir}")


if __name__ == "__main__":
    main()
