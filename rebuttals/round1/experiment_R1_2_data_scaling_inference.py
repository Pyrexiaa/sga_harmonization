"""Scoring the size sweep (manuscript Figure 7).

    Run experiment_R1_2_data_scaling.py first.

Run:
    python -m rebuttals.round1.experiment_R1_2_data_scaling_inference
    python -m rebuttals.round1.experiment_R1_2_data_scaling_inference --model catboost
"""

from __future__ import annotations

import argparse
import warnings

from sga.config import (
    ACCURACY_THRESHOLD,
    EXTERNAL_TEST_FOLD,
    HARMONIZED_SELECTED_FEATURES,
    N_FOLDS_CV,
    SEED,
    set_seed,
)
from sga.evaluation.country import append_fold_rows, evaluate_splits, save_country_results
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.model_io import load_trained_model, predict_labels_and_proba, weights_exist
from sga.pipeline.train_unified import build_harmonized_folds

from rebuttals.round1.experiment_R1_2_data_scaling import (
    DNN_CONFIG,
    MIN_SIZE,
    SAVE_DIR,
    SIZE_STEP,
    combined_original_size,
    scale_factor_for,
    size_label_for,
)

warnings.filterwarnings("ignore")

# model key -> (family, sub-directory template; {label} = size label, {seed} = seed)
MODELS = {
    "catboost": ("catboost", "catboost/size_{label}_{seed}"),
    "dnn": ("dnn", "dnn/size_{label}_{seed}"),
    "rf": ("ml", "ml/rf/size_{label}_{seed}"),
    "lr": ("ml", "ml/lr/size_{label}_{seed}"),
    "svc": ("ml", "ml/svc/size_{label}_{seed}"),
    "stacking": ("ml", "ml/stacking/size_{label}_{seed}"),
}


def build_scaled_external_folds(scale_factor, seed=SEED):
    """Rebuild the fold-4 external test set for one training-size condition."""
    set_seed(seed)
    msia_ds, india_ds = load_both_cohorts(exclude_external_fold=True)
    msia_full, india_full = load_both_cohorts(exclude_external_fold=False)

    folds = {}
    for prepared in build_harmonized_folds(
        msia_ds,
        india_ds,
        num_of_folds=N_FOLDS_CV,
        accuracy_threshold=ACCURACY_THRESHOLD,
        train_scale_factor=scale_factor,
        scale_seed=seed,
        selected_features=HARMONIZED_SELECTED_FEATURES,
        msia_ds_full=msia_full,
        india_ds_full=india_full,
        external_test_fold=EXTERNAL_TEST_FOLD,
    ):
        folds[prepared.fold] = (
            prepared.test_df[prepared.features].reset_index(drop=True),
            prepared.test_Y.to_numpy(dtype=int),
            prepared.country_arr,
            prepared.features,
        )
    return folds


def evaluate_model_size(model_key, family, subdir, target_size, original_size):
    """Score every fold-model of one (family, size) run on the external fold."""
    label = size_label_for(target_size, original_size)[0]
    download_path = SAVE_DIR / subdir.format(label=label, seed=SEED) / "malaysia_tri3"
    if not weights_exist(str(download_path), N_FOLDS_CV):
        return None

    factor = scale_factor_for(target_size, original_size)
    print(f"\n[{model_key}] size={label}  factor={factor:.4f}  ({download_path})")
    folds = build_scaled_external_folds(factor)

    rows = []
    for fold in range(N_FOLDS_CV):
        test_X, test_Y, country_arr, features = folds[fold]
        model = load_trained_model(
            family, str(download_path), fold,
            n_features=len(features), dnn_config=DNN_CONFIG,
        )
        if model is None:
            print(f"  [fold {fold}] weights missing -- skipped")
            continue

        y_pred, y_prob = predict_labels_and_proba(model, family, test_X)
        split_metrics = evaluate_splits(test_Y, y_pred, y_prob, country_arr)
        append_fold_rows(
            rows, split_metrics, model_key, fold,
            extra={"size": label, "train_scale_factor": round(factor, 4)},
        )
        print(
            f"  [fold-{fold} model -> fold-{EXTERNAL_TEST_FOLD} external] "
            f"MY AUC={split_metrics['malaysia']['roc_auc']:.4f} | "
            f"IN AUC={split_metrics['india']['roc_auc']:.4f} | "
            f"ALL AUC={split_metrics['total']['roc_auc']:.4f}"
        )

    if rows:
        save_country_results(
            rows, str(download_path), prefix="country",
            model_name=f"{model_key}/{label}",
            group_cols=("model", "size", "eval_split"),
        )
    return rows


def run_experiment(models=None):
    """Score every requested (family, size) combination that has weights."""
    models = list(MODELS) if models is None else list(models)
    msia_ds, india_ds = load_both_cohorts()
    original_size = combined_original_size(msia_ds, india_ds)
    all_sizes = sorted(
        set(list(range(MIN_SIZE, original_size, SIZE_STEP)) + [original_size]))
    print(f"Original combined size: {original_size}; sizes: {all_sizes}")

    all_rows = []
    for model_key in models:
        family, subdir = MODELS[model_key]
        for target_size in all_sizes:
            rows = evaluate_model_size(
                model_key, family, subdir, target_size, original_size)
            if rows:
                all_rows.extend(rows)

    if all_rows:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        save_country_results(
            all_rows, str(SAVE_DIR), prefix="per_country_eval_all",
            group_cols=("model", "size", "eval_split"),
        )
    else:
        print("\nNo trained weights found for the requested models/sizes.")
    return all_rows


def main():
    """Parse the command line and score the requested family."""
    parser = argparse.ArgumentParser(
        description="R1-2 no-retrain per-country evaluation of the size sweep")
    parser.add_argument("--model", default="all", choices=list(MODELS) + ["all"])
    args = parser.parse_args()
    set_seed(SEED)
    run_experiment(None if args.model == "all" else [args.model])


if __name__ == "__main__":
    set_seed(SEED)
    main()
