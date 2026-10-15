"""Train a country-specific baseline: one cohort, trained and tested on itself."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sga.config import MODEL_DIR, N_FOLDS_CV, SEED, set_seed
from sga.pipeline import train_baseline as baseline
from sga.pipeline.train_unified import SKLEARN_MODEL_TYPES

MODELS = (*SKLEARN_MODEL_TYPES, "catboost", "dnn")
DNN_HPARAMS = (0.20, 4, 100, 256, 1e-3, 5e-3, 0)


def main():
    """Seed, load one cohort and run the matching country-specific baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--country", choices=("malaysia", "india"), required=True)
    add("--model", choices=MODELS, default="lr")
    add("--output-dir", type=Path, default=None)
    add("--folds", type=int, default=None,
        help="cross-validation folds (default: 4, the development block)")
    add("--no-smote", action="store_true")
    add("--undersample", action="store_true")
    add("--pretrained", action="store_true")
    add("--skip-shap", action="store_true")
    args = parser.parse_args()
    folds = args.folds if args.folds is not None else N_FOLDS_CV
    out = args.output_dir or MODEL_DIR / f"baseline_{args.country}_{args.model}_{SEED}"

    set_seed(SEED)
    dfs, id_exists = baseline.load_country_dataset(args.country)
    common = dict(
        download_path=str(out), num_of_folds=folds, id_exists=id_exists,
        smoting=not args.no_smote, undersampling=args.undersample,
    )
    if args.model == "catboost":
        baseline.train_baseline_catboost(dfs, pretrained_model=args.pretrained, **common)
    elif args.model == "dnn":
        baseline.train_baseline_dnn(
            dfs, hyperparameters=DNN_HPARAMS, skip_shap=args.skip_shap, **common)
    else:
        baseline.train_baseline_sklearn(
            dfs, model_type=args.model, pretrained_model=args.pretrained, **common)
    print(f"Done. Results in {out}")


if __name__ == "__main__":
    main()
