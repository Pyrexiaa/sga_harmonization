"""Train the unified (pooled, harmonized) feed-forward neural network."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sga.config import ACCURACY_THRESHOLD, MODEL_DIR, N_FOLDS_CV, SEED, set_seed
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.train_unified import train_dnn_unified


def main():
    """Seed, load both cohorts and run the unified DNN training."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--output-dir", type=Path, default=MODEL_DIR / f"unified_dnn_{SEED}")
    add("--folds", type=int, default=N_FOLDS_CV)
    add("--accuracy-threshold", type=float, default=ACCURACY_THRESHOLD)
    add("--model-size", default="large")
    add("--hparams", type=float, nargs=7, default=(0.20, 4, 100, 256, 1e-3, 5e-3, 0))
    add("--train-source", choices=("malaysia", "india"), default=None)
    add("--drop-prev-pregnancy", action="store_true")
    add("--no-smote", action="store_true")
    add("--no-balanced-sampling", action="store_true")
    add("--skip-shap", action="store_true")
    add("--selected-features", nargs="*", default=None,
        help="explicit cross-domain feature selection, bypassing the imputation-"
             "quality gate. Pass with no values for a pooled COMMON-FEATURE-ONLY "
             "run (the unified_common_* arm of the Table 7 imputation ablation; Figure 3's three arms are baseline / unified / cross-domain).")
    args = parser.parse_args()
    dropout, width, epochs, batch, lr, decay, l1 = args.hparams

    set_seed(SEED)
    msia_ds, india_ds = load_both_cohorts()
    train_dnn_unified(
        msia_ds,
        india_ds,
        download_path=str(args.output_dir),
        hyperparameters=(dropout, int(width), int(epochs), int(batch), lr, decay, l1),
        num_of_folds=args.folds,
        model_size=args.model_size,
        accuracy_threshold=args.accuracy_threshold,
        smoting=not args.no_smote,
        pytorch_balanced_sampling=not args.no_balanced_sampling,
        skip_shap=args.skip_shap,
        drop_prev_pregnancy=args.drop_prev_pregnancy,
        train_source=args.train_source,
        selected_features=args.selected_features,
    )
    print(f"Done. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
