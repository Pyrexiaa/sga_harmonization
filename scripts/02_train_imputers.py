"""Train the per-feature cross-domain CatBoost imputation models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sga import config
from sga.config import set_seed
from sga.imputation.train_imputers import DEFAULT_TARGETS, train_all_imputers
from sga.pipeline.dataset import load_both_cohorts


def main():
    """Load both cohorts and train the requested cross-domain imputers."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--targets", nargs="*", choices=DEFAULT_TARGETS, default=None,
        help="features to model (default: every registered cross-domain feature)")
    add("--output-dir", type=Path, default=None)
    add("--chart", default=config.CHART)
    add("--impute-method", choices=("iterative", "mean", "median", "mode"),
        default="iterative", help="how missing model INPUTS are completed")
    add("--seed", type=int, default=config.SEED)
    add("--retrain", action="store_true",
        help="overwrite targets that already have weights")
    args = parser.parse_args()
    out_dir = args.output_dir or config.IMPUTER_DIR
    if Path(out_dir).resolve() != Path(config.IMPUTER_DIR).resolve():
        print(
            f"WARNING: writing imputers to {out_dir}, but the imputation-quality gate "
            f"in sga/imputation/apply.py reads {config.IMPUTER_DIR}. The downstream "
            "training scripts will report 'no metrics ...' and drop every cross-domain "
            "feature. Set SGA_PROJECT_ROOT (or SGA_SEED) instead of --output-dir to "
            "move the whole tree consistently."
        )

    set_seed(args.seed)
    # The external test fold is excluded here as well; the imputers are part of the
    # pipeline and must never see fold 4.
    msia_ds, india_ds = load_both_cohorts(chart=args.chart)
    trained = train_all_imputers(
        {"malaysia": msia_ds, "india": india_ds},
        targets=args.targets,
        base_dir=out_dir,
        impute_method=args.impute_method,
        seed=args.seed,
        skip_if_trained=not args.retrain,
    )
    print(f"\nTrained {len(trained)} imputer(s): {trained or 'none (all up to date)'}")
    print(f"Weights and metrics in {out_dir}")


if __name__ == "__main__":
    main()
