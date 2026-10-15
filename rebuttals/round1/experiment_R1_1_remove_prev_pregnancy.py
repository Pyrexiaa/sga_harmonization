"""Ablate the India-only maternal-history features.

    Run experiment_R0_baseline_retrain.py first (the pretrained CatBoost
    imputers, and the with-maternal-history baseline these results are compared
    against).

Run:
    python -m rebuttals.round1.experiment_R1_1_remove_prev_pregnancy
"""

from __future__ import annotations

from sga.config import (
    ACCURACY_THRESHOLD,
    PREV_PREGNANCY_FEATURES,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.train_unified import (
    train_catboost_unified,
    train_dnn_unified,
    train_sklearn_unified,
)

SAVE_DIR = ROUND1_DIR / "R1_1_remove_prev_pregnancy"

DNN_CONFIGS = [
    (0.20, 4, 100, 256, 1e-3, 5e-3, 0),
    (0.20, 4, 100, 256, 1e-3, 1e-3, 0),
    (0.20, 6, 100, 256, 1e-3, 5e-3, 0),
]
ML_MODELS = ["rf", "lr", "svc", "stacking"]
ML_HYPERPARAMS = (0.20, 4, 200, 256, 5e-3, 5e-3, 0)


def drop_prev_pregnancy_features(india_ds):
    """Return a copy of the India dataset pair without the maternal-history columns."""
    india_df, india_add_on = india_ds[0].copy(), india_ds[1].copy()
    for feature in PREV_PREGNANCY_FEATURES:
        india_df = india_df.drop(columns=[feature], errors="ignore")
        india_add_on = india_add_on.drop(columns=[feature], errors="ignore")
    print(f"  Dropped from India df: "
          f"{[f for f in PREV_PREGNANCY_FEATURES if f not in india_df.columns]}")
    return [india_df, india_add_on]


def run_catboost_experiment():
    """Retrain the unified CatBoost model without the maternal-history features."""
    print("\nR1-1 CATBOOST: unified pipeline without previous-pregnancy features")
    msia_ds, india_ds = load_both_cohorts()
    india_no_prev = drop_prev_pregnancy_features(india_ds)

    save_dir = SAVE_DIR / "catboost" / f"without_prev_preg_{SEED}"
    train_catboost_unified(
        msia_ds,
        india_no_prev,
        download_path=str(save_dir / "malaysia_tri3"),
        smoting=True,
        accuracy_threshold=ACCURACY_THRESHOLD,
        drop_prev_pregnancy=True,
    )
    print(f"\n[R1-1] CatBoost complete.\n  Ablation:  {save_dir}")


def run_ml_experiment():
    """Retrain each classical classifier without the maternal-history features."""
    print("\nR1-1 ML: unified pipeline without previous-pregnancy features")
    for model_type in ML_MODELS:
        print(f"\nML model: {model_type}")
        msia_ds, india_ds = load_both_cohorts()
        india_no_prev = drop_prev_pregnancy_features(india_ds)

        save_dir = SAVE_DIR / "ml" / model_type / f"without_prev_preg_{SEED}"
        train_sklearn_unified(
            msia_ds,
            india_no_prev,
            download_path=str(save_dir / "malaysia_tri3"),
            model_type=model_type,
            smoting=True,
            accuracy_threshold=ACCURACY_THRESHOLD,
            drop_prev_pregnancy=True,
        )
        print(f"  {model_type} ablation:  {save_dir}")
    print("\n[R1-1] ML experiment complete.")


def run_dnn_experiment():
    """Retrain the neural network under each configuration without those features."""
    print("\nR1-1 DNN: unified pipeline without previous-pregnancy features")
    for index, config in enumerate(DNN_CONFIGS):
        print(f"\nDNN config {index}: {config}")
        msia_ds, india_ds = load_both_cohorts()
        india_no_prev = drop_prev_pregnancy_features(india_ds)

        save_dir = SAVE_DIR / "dnn" / f"config_{index}" / f"without_prev_preg_{SEED}"
        train_dnn_unified(
            msia_ds,
            india_no_prev,
            download_path=str(save_dir / "malaysia_tri3"),
            hyperparameters=config,
            smoting=True,
            accuracy_threshold=ACCURACY_THRESHOLD,
            pytorch_balanced_sampling=True,
            drop_prev_pregnancy=True,
        )
        print(f"  DNN config {index} ablation:  {save_dir}")
    print("\n[R1-1] DNN experiment complete.")


def run_experiment():
    """Run the CatBoost, classical and neural ablations in that order."""
    print("R1-1: REMOVE PREVIOUS PREGNANCY FEATURES EXPERIMENT")
    print(f"    Seed: {SEED}")
    print(f"    Features removed: {PREV_PREGNANCY_FEATURES}")
    print(f"    Output: {SAVE_DIR}")

    run_catboost_experiment()
    run_ml_experiment()
    run_dnn_experiment()

    print(f"\nR1-1: ALL EXPERIMENTS COMPLETE\n    Results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
