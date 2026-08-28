"""Build the Malaysia training/testing tables (Methods, "Study Population").

Usage:
    python -m scripts.01a_prepare_malaysia
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sga import config
from sga.config import set_seed
from sga.data.cleaning import remove_illogical_values
from sga.data.preprocess_malaysia import (
    RAW_SUBPATH,
    drop_postnatal_columns,
    manual_remove_rare_cases,
    preprocess_malaysia,
)
from sga.data.splits import split_complete_and_addon


def main():
    """Screen the Malaysian cohort, assign folds and write the training tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--raw-csv", type=Path, default=config.RAW_DATA_DIR / RAW_SUBPATH)
    add("--output-dir", type=Path, default=None)
    add("--chart", default=config.CHART)
    add("--folds", type=int, default=config.N_FOLDS_TOTAL)
    add("--seed", type=int, default=config.SEED)
    add("--drop-columns", nargs="*", default=None,
        help="extra raw columns to discard, e.g. --drop-columns cpr")
    args = parser.parse_args()
    out_dir = args.output_dir or config.TRAINING_DATA_DIR / config.MALAYSIA_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    df, log = preprocess_malaysia(
        args.raw_csv, chart=args.chart, drop_columns=args.drop_columns
    )
    df = log.apply("implausible on every biometric measurement",
                   manual_remove_rare_cases(df))

    # Postnatal fields are dropped before the split so they can never reach a model;
    # ``birth_ga`` goes too, it is only needed for the centile lookup.
    df = drop_postnatal_columns(df, extra=["birth_ga"])
    remove_illogical_values(df, keep_nan=True)
    log.record("measurement outside physiological range (post-screening)", len(df))

    complete, add_on = split_complete_and_addon(
        df,
        categorical_col=None,
        num_folds=args.folds,
        label=config.LABEL,
        id_exist=True,
        seed=args.seed,
    )

    complete.to_csv(out_dir / f"tri3_{args.chart}.csv", index=False)
    add_on.to_csv(out_dir / f"tri3_add_on_{args.chart}.csv", index=False)
    log.to_frame().to_csv(out_dir / f"consort_flow_{args.chart}.csv", index=False)

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
