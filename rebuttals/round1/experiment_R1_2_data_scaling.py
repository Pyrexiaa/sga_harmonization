"""Training-set-size scaling (manuscript Figure 7).

    Run experiment_R0_baseline_retrain.py first (the pretrained CatBoost
    imputers). Score the resulting weights with
    experiment_R1_2_data_scaling_inference.py.

Run:
    python -m rebuttals.round1.experiment_R1_2_data_scaling
    python -m rebuttals.round1.experiment_R1_2_data_scaling --model svc --workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from sga.config import (
    ACCURACY_THRESHOLD,
    HARMONIZED_SELECTED_FEATURES,
    LABEL,
    N_FOLDS_CV,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.model_io import weights_exist
from sga.pipeline.train_unified import (
    train_catboost_unified,
    train_dnn_unified,
    train_sklearn_unified,
)

SAVE_DIR = ROUND1_DIR / "R1_2_data_scaling"

# Target combined training sizes:
MIN_SIZE = 1000
SIZE_STEP = 1000

FORCE_RETRAIN = False  # set True to retrain even if weights already exist

# Representative DNN configuration (config 0 from the paper).
DNN_CONFIG = (0.20, 4, 100, 256, 1e-3, 5e-3, 0)
ML_MODELS = ("rf", "lr", "svc", "stacking")


def combined_original_size(msia_ds, india_ds):
    """Full combined size: Malaysia and India, complete plus add-on records."""
    return len(msia_ds[0]) + len(msia_ds[1]) + len(india_ds[0]) + len(india_ds[1])


def scale_factor_for(target_size, original_size):
    """Training scaling factor for a target size (1.0 at or above the original)."""
    return 1.0 if target_size >= original_size else target_size / original_size


def size_label_for(target_size, original_size):
    """Directory label for a target size, and whether it is the original size."""
    is_original = target_size >= original_size
    return (f"original_{original_size}" if is_original else str(target_size)), is_original


def _skip(save_dir, label):
    """True when this size already has a complete set of fold weights."""
    if not FORCE_RETRAIN and weights_exist(str(save_dir), N_FOLDS_CV):
        print(f"[skip] weights present, no retrain: {save_dir}/model_weights")
        return True
    del label
    return False


def run_catboost_one_size(target_size, original_size):
    """Train the unified CatBoost model at one combined target size."""
    set_seed(SEED)
    msia_ds, india_ds = load_both_cohorts()
    factor = scale_factor_for(target_size, original_size)
    label, is_original = size_label_for(target_size, original_size)
    print(f"[CatBoost] target={target_size} {'(ORIGINAL)' if is_original else ''} "
          f"train_scale_factor={factor:.4f}")

    save_dir = SAVE_DIR / "catboost" / f"size_{label}_{SEED}" / "malaysia_tri3"
    if _skip(save_dir, "catboost"):
        return f"{save_dir} skipped (weights exist)"
    train_catboost_unified(
        msia_ds, india_ds, download_path=str(save_dir),
        smoting=True, accuracy_threshold=ACCURACY_THRESHOLD,
        train_scale_factor=factor, scale_seed=SEED,
        selected_features=HARMONIZED_SELECTED_FEATURES,
    )
    return f"CatBoost target={target_size} done"


def run_ml_one_size(target_size, original_size, model_type):
    """Train one classical classifier at one combined target size."""
    set_seed(SEED)
    msia_ds, india_ds = load_both_cohorts()
    factor = scale_factor_for(target_size, original_size)
    label, is_original = size_label_for(target_size, original_size)
    print(f"[{model_type.upper()}] target={target_size} "
          f"{'(ORIGINAL)' if is_original else ''} train_scale_factor={factor:.4f}")

    save_dir = SAVE_DIR / "ml" / model_type / f"size_{label}_{SEED}" / "malaysia_tri3"
    if _skip(save_dir, model_type):
        return f"{save_dir} skipped (weights exist)"
    train_sklearn_unified(
        msia_ds, india_ds, download_path=str(save_dir), model_type=model_type,
        smoting=True, accuracy_threshold=ACCURACY_THRESHOLD,
        train_scale_factor=factor, scale_seed=SEED,
        selected_features=HARMONIZED_SELECTED_FEATURES,
    )
    return f"{model_type.upper()} target={target_size} done"


def run_dnn_one_size(target_size, original_size):
    """Train the unified neural network at one combined target size."""
    set_seed(SEED)
    msia_ds, india_ds = load_both_cohorts()
    factor = scale_factor_for(target_size, original_size)
    label, is_original = size_label_for(target_size, original_size)
    print(f"[DNN] target={target_size} {'(ORIGINAL)' if is_original else ''} "
          f"train_scale_factor={factor:.4f}")

    save_dir = SAVE_DIR / "dnn" / f"size_{label}_{SEED}" / "malaysia_tri3"
    if _skip(save_dir, "dnn"):
        return f"{save_dir} skipped (weights exist)"
    train_dnn_unified(
        msia_ds, india_ds, download_path=str(save_dir), hyperparameters=DNN_CONFIG,
        smoting=True, accuracy_threshold=ACCURACY_THRESHOLD,
        pytorch_balanced_sampling=True, skip_shap=True,
        train_scale_factor=factor, scale_seed=SEED,
        selected_features=HARMONIZED_SELECTED_FEATURES,
    )
    return f"DNN target={target_size} done"


def run_parallel(function, all_sizes, original_size, max_workers, label, **kwargs):
    """Run one training function across every target size."""
    print(f"\nR1-2 {label}: data scaling ({len(all_sizes)} sizes, {max_workers} workers)")
    if max_workers <= 1:
        for size in all_sizes:
            print(function(size, original_size, **kwargs))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(function, size, original_size, **kwargs): size
                for size in all_sizes
            }
            for future in as_completed(futures):
                size = futures[future]
                try:
                    print(future.result())
                except Exception as error:  # noqa: BLE001
                    print(f"[ERROR] size={size}: {error}")
    print(f"\n[R1-2] {label} scaling complete.")


def run_experiment(model="catboost", workers=1):
    """Train the requested family across the whole size sweep."""
    msia_ds, india_ds = load_both_cohorts()
    malaysia_total = len(msia_ds[0]) + len(msia_ds[1])
    india_total = len(india_ds[0]) + len(india_ds[1])
    original_size = malaysia_total + india_total

    combined = pd.concat([msia_ds[0], msia_ds[1], india_ds[0], india_ds[1]])
    all_sizes = sorted(
        set(list(range(MIN_SIZE, original_size, SIZE_STEP)) + [original_size]))

    print("R1-2: DATA SCALING EXPERIMENT (unified pipeline)")
    print(f"    Seed: {SEED}")
    print(f"    Malaysia (complete + add-on): {len(msia_ds[0])} + {len(msia_ds[1])} "
          f"= {malaysia_total}")
    print(f"    India    (complete + add-on): {len(india_ds[0])} + {len(india_ds[1])} "
          f"= {india_total}")
    print(f"    Original COMBINED size:   {original_size}")
    print(f"    Malaysia fraction:        {malaysia_total / original_size:.4f}")
    print(f"    Combined SGA ratio:       {combined[LABEL].mean():.4f}")
    print(f"    Target sizes: {all_sizes}")
    print(f"    Model: {model}    Workers: {workers}")
    print(f"    Output: {SAVE_DIR}")

    if model in ("catboost", "all"):
        run_parallel(run_catboost_one_size, all_sizes, original_size,
                     max_workers=workers, label="CATBOOST")
    for model_type in ML_MODELS:
        if model in (model_type, "all"):
            run_parallel(run_ml_one_size, all_sizes, original_size,
                         max_workers=workers, label=model_type.upper(),
                         model_type=model_type)
    if model in ("dnn", "all"):
        run_parallel(run_dnn_one_size, all_sizes, original_size,
                     max_workers=1 if model == "all" else workers, label="DNN")

    print(f"\nR1-2: ALL SCALING EXPERIMENTS COMPLETE\n    Results saved to: {SAVE_DIR}")


def main():
    """Parse the command line and run the requested sweep."""
    parser = argparse.ArgumentParser(description="R1-2: training-set-size scaling")
    parser.add_argument(
        "--model", default="catboost",
        choices=["catboost", "dnn", *ML_MODELS, "all"],
        help="Which model family to train. Run separate terminals for parallelism.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Target sizes trained concurrently. Not recommended for GPU models.",
    )
    args = parser.parse_args()
    set_seed(SEED)
    run_experiment(model=args.model, workers=args.workers)


if __name__ == "__main__":
    set_seed(SEED)
    main()
