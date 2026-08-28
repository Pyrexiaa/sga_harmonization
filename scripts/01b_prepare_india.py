"""Build the India training/testing tables (Methods, "Study Population").

Usage:
    python -m scripts.01b_prepare_india
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sga import config
from sga.config import set_seed
from sga.data.cleaning import remove_illogical_values
from sga.data.preprocess_india import (
    RAW_APRIL8_SUBPATH,
    RAW_MAY8_SUBPATH,
    cast_integer_columns,
    combine_india_drops,
    drop_postnatal_columns,
    preprocess_india_april8,
    preprocess_india_may8,
)
from sga.data.screening import ScreeningLog
from sga.data.splits import split_complete_and_addon


def main():
    """Screen both Indian drops, merge them, assign folds and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--may8-csv", type=Path, default=config.RAW_DATA_DIR / RAW_MAY8_SUBPATH)
    add("--april8-csv", type=Path, default=config.RAW_DATA_DIR / RAW_APRIL8_SUBPATH)
    add("--output-dir", type=Path, default=None)
    add("--chart", default=config.CHART)
    add("--folds", type=int, default=config.N_FOLDS_TOTAL)
    add("--seed", type=int, default=config.SEED)
    add("--drop-columns", nargs="*", default=None,
        help="extra raw columns to discard from both drops, e.g. --drop-columns cpr. "
             "Empty by default: cpr is one of the ten COMMON_FEATURES and a required "
             "input to every cross-domain imputer, so dropping it breaks the pipeline.")
    args = parser.parse_args()
    out_dir = args.output_dir or config.TRAINING_DATA_DIR / config.INDIA_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    may8, may8_log = preprocess_india_may8(
        args.may8_csv, chart=args.chart, drop_columns=args.drop_columns
    )
    april8, april8_log = preprocess_india_april8(
        args.april8_csv, chart=args.chart, drop_columns=args.drop_columns
    )

    screened = may8_log.screened + april8_log.screened
    log = ScreeningLog("India (combined)", screened)
    log.record("excluded during per-drop screening", len(may8) + len(april8))
    df, log = combine_india_drops(april8, may8, log=log)

    df = drop_postnatal_columns(df)
    df = cast_integer_columns(df)
    remove_illogical_values(df, keep_nan=True)
    log.record("measurement outside physiological range", len(df))

    complete, add_on = split_complete_and_addon(
        df,
        categorical_col=None,
        num_folds=args.folds,
        label=config.LABEL,
        id_exist=False,
        seed=args.seed,
    )
    for frame in (complete, add_on):
        cast_integer_columns(frame)

    complete.to_csv(out_dir / f"tri3_{args.chart}.csv", index=False)
    add_on.to_csv(out_dir / f"tri3_add_on_{args.chart}.csv", index=False)
    pd.concat(
        [may8_log.to_frame(), april8_log.to_frame(), log.to_frame()], ignore_index=True
    ).to_csv(out_dir / f"consort_flow_{args.chart}.csv", index=False)

    may8_log.report()
    april8_log.report()

    entering = len(complete) + len(add_on)
    positives = int(complete[config.LABEL].sum() + add_on[config.LABEL].sum())
    log.report(
        entering,
        extra={
            "  complete-case records (folds 0-4)": len(complete),
            "  add-on records (fold -1, training only)": len(add_on),
            "  SGA-positive records": positives,
            f"  held-out external fold {config.EXTERNAL_TEST_FOLD}":
                int((complete["fold"] == config.EXTERNAL_TEST_FOLD).sum()),
        },
    )
    print(f"\nWrote tri3_{args.chart}.csv and tri3_add_on_{args.chart}.csv to {out_dir}")


if __name__ == "__main__":
    main()
