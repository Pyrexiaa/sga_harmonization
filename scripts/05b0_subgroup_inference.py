"""Per-row re-inference on the unified external test set (Figure 5 prerequisite).

Run once per arm, e.g.::

    python -m scripts.05b0_subgroup_inference --family ml --model lr \
        --model-dir <results>/models/unified_lr_123 --name unified
    python -m scripts.05b0_subgroup_inference --family ml --model lr \
        --model-dir <results>/models/baseline_malaysia_lr_123 \
        --train-source malaysia --baseline-features --name malaysia_baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sga import config
from sga.config import set_seed
from sga.data.scaling import descale_feature
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.inference import CATBOOST, DNN, SKLEARN, load_model, predict
from sga.pipeline.train_baseline import BASELINE_COMMON_FEATURES
from sga.pipeline.train_unified import build_harmonized_folds

FAMILIES = (CATBOOST, SKLEARN, DNN)

#: Representative DNN configuration (dropout, layer width, ...) used in the paper.
DNN_HPARAMS = (0.20, 4, 100, 256, 1e-3, 5e-3, 0)

#: Cohort code -> the label written into the ``country`` column.
COUNTRY_NAMES = {0: "malaysia", 1: "india"}


def descaled_features(fold):
    """Return the fold's scaled test matrix converted back to clinical units."""
    raw = fold.test_df[fold.features].copy()
    continuous = [c for c in fold.continuous_features if c in raw.columns]
    raw[continuous] = descale_feature(raw[continuous], fold.std_or_min, fold.mean_or_max)
    return raw


def save_predictions(path, features, y_true, y_pred, y_prob, country_arr):
    """Write one prediction CSV in the layout the Figure 5 script expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = features.reset_index(drop=True).copy()
    out["Actual"] = np.asarray(y_true)
    out["Prediction"] = np.asarray(y_pred)
    out["predicted_probability"] = np.asarray(y_prob)
    out["country"] = [COUNTRY_NAMES[int(c)] for c in country_arr]
    out.to_csv(path, index=False)
    print(f"  wrote {path} ({len(out)} rows)")


def main():
    """Re-score every fold-model of one arm on the unified external test set."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--family", choices=FAMILIES, required=True)
    add("--model", required=True, help="model key used in the output file names")
    add("--model-dir", type=Path, required=True,
        help="training run directory holding model_weights/model_<fold>")
    add("--name", default=None, help="output subdirectory name (default: --model)")
    add("--output-dir", type=Path, default=None)
    add("--train-source", choices=("malaysia", "india"), default=None,
        help="set for the country-specific baseline arms")
    add("--accuracy-threshold", type=float, default=config.ACCURACY_THRESHOLD)
    add("--folds", type=int, default=config.N_FOLDS_CV)
    add("--external-fold", type=int, default=config.EXTERNAL_TEST_FOLD)
    add("--chart", default=config.CHART)
    add("--seed", type=int, default=config.SEED)
    add("--model-size", default="large")
    add("--drop-prev-pregnancy", action="store_true")
    add("--no-smote", action="store_true")
    add("--with-validation", action="store_true",
        help="set when the arm was trained with a held-out validation fold. Required "
             "for the unified DNN; implied automatically by --baseline-features, "
             "because train_baseline always holds one fold out.")
    add("--selected-features", nargs="*", default=None,
        help="explicit cross-domain feature selection, matching what the arm was "
             "trained under. Pass with no values for a common-feature-only arm.")
    add("--baseline-features", action="store_true",
        help="restrict the model matrix to BASELINE_COMMON_FEATURES (the ten "
             "shared measurements). "
             "Required for the country-baseline arms, whose weights were fitted "
             "on that narrower space and cannot be loaded into a harmonized fold.")
    args = parser.parse_args()

    out_dir = Path(
        args.output_dir
        or config.results_path("figures", "figure5", "predictions")
    ) / (args.name or args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    msia_ds, india_ds = load_both_cohorts(chart=args.chart, exclude_external_fold=True)
    msia_full, india_full = load_both_cohorts(
        chart=args.chart, exclude_external_fold=False
    )

    probabilities, template, truth, countries = [], None, None, None
    for fold in build_harmonized_folds(
        msia_ds,
        india_ds,
        num_of_folds=args.folds,
        label=config.LABEL,
        accuracy_threshold=args.accuracy_threshold,
        drop_prev_pregnancy=args.drop_prev_pregnancy,
        train_source=args.train_source,
        smoting=not args.no_smote,
        # `train_baseline.build_baseline_folds` unconditionally holds fold (i+1)%n out
        # of the training rows, so a baseline arm must be rebuilt the same way or the
        # test matrix is standardised with statistics the weights never saw.
        with_validation=args.with_validation or args.baseline_features,
        msia_ds_full=msia_full,
        india_ds_full=india_full,
        external_test_fold=args.external_fold,
        selected_features=(
            [] if args.baseline_features and args.selected_features is None
            else args.selected_features
        ),
        feature_subset=BASELINE_COMMON_FEATURES if args.baseline_features else None,
        imputer_seed=args.seed,
    ):
        model = load_model(
            args.family,
            str(args.model_dir),
            fold.fold,
            n_features=len(fold.features),
            dnn_config=DNN_HPARAMS,
            model_size=args.model_size,
        )
        if model is None:
            print(f"  [fold {fold.fold}] weights missing in {args.model_dir}; skipped")
            continue

        y_true = fold.test_df[config.LABEL].values.astype(int)
        y_pred, y_prob = predict(model, args.family, fold.test_df[fold.features])
        template = descaled_features(fold)
        truth, countries = y_true, fold.country_arr
        probabilities.append(y_prob)
        save_predictions(
            out_dir / f"fold_{fold.fold}.csv",
            template, y_true, y_pred, y_prob, fold.country_arr,
        )

    if not probabilities:
        raise SystemExit(
            f"No fold-models could be scored under {args.model_dir}; run the "
            "matching 03* training script first."
        )

    ensemble = np.mean(np.vstack(probabilities), axis=0)
    save_predictions(
        out_dir / "ensemble.csv",
        template, truth, (ensemble >= config.DECISION_THRESHOLD).astype(int),
        ensemble, countries,
    )
    summary = pd.DataFrame({"country": [COUNTRY_NAMES[int(c)] for c in countries]})
    print(
        f"\nWrote {len(probabilities)} fold CSVs plus ensemble.csv to {out_dir}\n"
        f"Rows per cohort: {summary['country'].value_counts().to_dict()}"
    )


if __name__ == "__main__":
    main()
