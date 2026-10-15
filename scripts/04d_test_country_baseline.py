"""Score a country-specific baseline fold-model set on the held-out external fold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sga import config
from sga.config import set_seed
from sga.pipeline.inference import CATBOOST, DNN, SKLEARN, run_external_inference
from sga.pipeline.train_baseline import BASELINE_COMMON_FEATURES
from sga.pipeline.train_unified import SKLEARN_MODEL_TYPES

MODELS = (*SKLEARN_MODEL_TYPES, "catboost", "dnn")
DNN_HPARAMS = (0.20, 4, 100, 256, 1e-3, 5e-3, 0)

FAMILIES = {"catboost": CATBOOST, "dnn": DNN}


def main():
    """Score one country baseline's fold-models on the external test fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    seed = config.SEED
    add("--country", choices=("malaysia", "india"), required=True)
    add("--model", choices=MODELS, default="lr")
    add("--model-dir", type=Path, default=None)
    add("--folds", type=int, default=config.N_FOLDS_CV)
    add("--external-fold", type=int, default=config.EXTERNAL_TEST_FOLD)
    add("--model-size", default="large")
    add("--no-smote", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir or (
        config.MODEL_DIR / f"baseline_{args.country}_{args.model}_{seed}"
    )
    family = FAMILIES.get(args.model, SKLEARN)

    set_seed(seed)
    run_external_inference(
        family,
        args.model,
        download_path=str(model_dir),
        num_of_folds=args.folds,
        train_source=args.country,
        smoting=not args.no_smote,
        external_test_fold=args.external_fold,
        # The baseline never used cross-domain imputation, and it only ever saw the ten
        # shared measurements.
        selected_features=[],
        feature_subset=BASELINE_COMMON_FEATURES,
        # `train_baseline.build_baseline_folds` ALWAYS holds fold (i+1)%n out of the
        # training rows, so the standardiser behind the saved weights was fitted
        # without it. Rebuilding the fold without `with_validation` here would scale
        # the fold-4 test matrix with different statistics than the model was fitted
        # under and shift every baseline prediction.
        with_validation=True,
        dnn_config=DNN_HPARAMS if args.model == "dnn" else None,
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()
