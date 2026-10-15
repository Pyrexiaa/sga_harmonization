"""Train the unified (pooled, harmonized) CatBoost classifier.

Usage:
    python -m scripts.03a_train_unified_catboost
    python scripts/03a_train_unified_catboost.py --accuracy-threshold 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sga.config import ACCURACY_THRESHOLD, MODEL_DIR, N_FOLDS_CV, SEED, set_seed
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.train_unified import train_catboost_unified


def main():
    """Seed, load both cohorts and run the unified CatBoost training."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--output-dir", type=Path, default=MODEL_DIR / f"unified_catboost_{SEED}")
    add("--folds", type=int, default=N_FOLDS_CV)
    add("--accuracy-threshold", type=float, default=ACCURACY_THRESHOLD)
    add("--train-source", choices=("malaysia", "india"), default=None)
    add("--train-scale-factor", type=float, default=1.0)
    add("--drop-prev-pregnancy", action="store_true")
    add("--no-smote", action="store_true")
    add("--undersample", action="store_true")
    add("--selected-features", nargs="*", default=None,
        help="explicit cross-domain feature selection, bypassing the imputation-"
             "quality gate. Pass with no values for a pooled COMMON-FEATURE-ONLY "
             "run (the unified_common_* arm of the Table 7 imputation ablation; Figure 3's three arms are baseline / unified / cross-domain).")
    args = parser.parse_args()

    set_seed(SEED)
    msia_ds, india_ds = load_both_cohorts()
    train_catboost_unified(
        msia_ds,
        india_ds,
        download_path=str(args.output_dir),
        num_of_folds=args.folds,
        accuracy_threshold=args.accuracy_threshold,
        smoting=not args.no_smote,
        undersampling=args.undersample,
        drop_prev_pregnancy=args.drop_prev_pregnancy,
        train_source=args.train_source,
        train_scale_factor=args.train_scale_factor,
        scale_seed=SEED,
        selected_features=args.selected_features,
    )
    print(f"Done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
