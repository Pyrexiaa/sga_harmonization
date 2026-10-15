"""Score the unified DNN fold-models on the held-out external fold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sga import config
from sga.config import set_seed
from sga.pipeline.inference import DNN, run_external_inference


def main():
    """Seed and run the held-out external-fold inference for the DNN."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    seed = config.SEED
    add("--model-dir", type=Path, default=config.MODEL_DIR / f"unified_dnn_{seed}")
    add("--folds", type=int, default=config.N_FOLDS_CV)
    add("--accuracy-threshold", type=float, default=config.ACCURACY_THRESHOLD)
    add("--external-fold", type=int, default=config.EXTERNAL_TEST_FOLD)
    add("--model-size", default="large")
    add("--dropout", type=float, default=0.20)
    add("--width", type=int, default=4)
    add("--train-source", choices=("malaysia", "india"), default=None)
    add("--drop-prev-pregnancy", action="store_true")
    add("--no-smote", action="store_true")
    add("--selected-features", nargs="*", default=None,
        help="explicit cross-domain feature selection, bypassing the imputation-"
             "quality gate. Pass with no values for a pooled COMMON-FEATURE-ONLY "
             "run (the unified_common_* arm of the Table 7 imputation ablation; Figure 3's three arms are baseline / unified / cross-domain).")
    args = parser.parse_args()

    set_seed(seed)
    run_external_inference(
        DNN,
        "dnn",
        download_path=str(args.model_dir),
        num_of_folds=args.folds,
        accuracy_threshold=args.accuracy_threshold,
        drop_prev_pregnancy=args.drop_prev_pregnancy,
        train_source=args.train_source,
        smoting=not args.no_smote,
        with_validation=True,
        dnn_config=(args.dropout, args.width),
        model_size=args.model_size,
        external_test_fold=args.external_fold,
        selected_features=args.selected_features,
    )


if __name__ == "__main__":
    main()
