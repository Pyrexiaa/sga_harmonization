"""Score a unified classical fold-model set on the held-out external fold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sga import config
from sga.config import set_seed
from sga.pipeline.inference import SKLEARN, run_external_inference
from sga.pipeline.train_unified import SKLEARN_MODEL_TYPES


def main():
    """Seed and run the held-out external-fold inference for one ML model."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    seed = config.SEED
    add("--model", choices=SKLEARN_MODEL_TYPES, default="lr")
    add("--model-dir", type=Path, default=None)
    add("--folds", type=int, default=config.N_FOLDS_CV)
    add("--accuracy-threshold", type=float, default=config.ACCURACY_THRESHOLD)
    add("--external-fold", type=int, default=config.EXTERNAL_TEST_FOLD)
    add("--train-source", choices=("malaysia", "india"), default=None)
    add("--drop-prev-pregnancy", action="store_true")
    add("--no-smote", action="store_true")
    add("--selected-features", nargs="*", default=None,
        help="explicit cross-domain feature selection, bypassing the imputation-"
             "quality gate. Pass with no values for a pooled COMMON-FEATURE-ONLY "
             "run (the unified_common_* arm of the Table 7 imputation ablation; Figure 3's three arms are baseline / unified / cross-domain).")
    args = parser.parse_args()
    model_dir = args.model_dir or config.MODEL_DIR / f"unified_{args.model}_{seed}"

    set_seed(seed)
    run_external_inference(
        SKLEARN,
        args.model,
        download_path=str(model_dir),
        num_of_folds=args.folds,
        accuracy_threshold=args.accuracy_threshold,
        drop_prev_pregnancy=args.drop_prev_pregnancy,
        train_source=args.train_source,
        smoting=not args.no_smote,
        external_test_fold=args.external_fold,
        selected_features=args.selected_features,
    )


if __name__ == "__main__":
    main()
