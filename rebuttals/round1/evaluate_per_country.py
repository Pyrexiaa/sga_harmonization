"""Per-cohort evaluation of already-trained unified models -- no retraining.

    Run experiment_R0_baseline_retrain.py and
    experiment_R1_1_remove_prev_pregnancy.py first.

Run:
    python -m rebuttals.round1.evaluate_per_country
"""

from __future__ import annotations

import warnings

from sga.config import (
    ACCURACY_THRESHOLD,
    LABEL,
    N_FOLDS_CV,
    PREV_PREGNANCY_FEATURES,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.evaluation.country import append_fold_rows, evaluate_splits, save_country_results
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.model_io import load_trained_model, predict_labels_and_proba
from sga.pipeline.train_unified import build_harmonized_folds

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND1_DIR / "per_country_no_retrain"
R0_DIR = ROUND1_DIR / "R0_baseline_retrain"
R1_1_DIR = ROUND1_DIR / "R1_1_remove_prev_pregnancy"

DNN_CONFIGS = [
    (0.20, 4, 100, 256, 1e-3, 5e-3, 0),
    (0.20, 4, 100, 256, 1e-3, 1e-3, 0),
    (0.20, 6, 100, 256, 1e-3, 5e-3, 0),
]
ML_MODELS = ["rf", "lr", "svc", "stacking", "stacking_catboost"]

VARIANTS = ("with_prev_preg", "without_prev_preg")


def _build_targets():
    """Every (name, family, variant, weights path) combination to evaluate."""
    targets = [
        dict(name="catboost_with", family="catboost", variant="with_prev_preg",
             path=str(R0_DIR / f"generalized_catboost_{SEED}" / "malaysia_tri3")),
    ]
    for model_type in ML_MODELS:
        targets.append(dict(
            name=f"{model_type}_with", family="ml", variant="with_prev_preg",
            path=str(R0_DIR / f"generalized_{model_type}_{SEED}" / "malaysia_tri3")))
    for index, config in enumerate(DNN_CONFIGS):
        targets.append(dict(
            name=f"dnn_cfg{index}_with", family="dnn", variant="with_prev_preg",
            dnn_config=config,
            path=str(R0_DIR / f"generalized_dnn_{SEED}_{index}" / "malaysia_tri3")))

    targets.append(dict(
        name="catboost_without", family="catboost", variant="without_prev_preg",
        path=str(R1_1_DIR / "catboost" / f"without_prev_preg_{SEED}" / "malaysia_tri3")))
    for model_type in ML_MODELS:
        targets.append(dict(
            name=f"{model_type}_without", family="ml", variant="without_prev_preg",
            path=str(R1_1_DIR / "ml" / model_type / f"without_prev_preg_{SEED}"
                     / "malaysia_tri3")))
    for index, config in enumerate(DNN_CONFIGS):
        targets.append(dict(
            name=f"dnn_cfg{index}_without", family="dnn", variant="without_prev_preg",
            dnn_config=config,
            path=str(R1_1_DIR / "dnn" / f"config_{index}" / f"without_prev_preg_{SEED}"
                     / "malaysia_tri3")))
    return targets


TARGETS = _build_targets()


def load_variant_datasets(drop_prev_pregnancy):
    """Load both cohorts, optionally without the India maternal-history columns."""
    msia_ds, india_ds = load_both_cohorts()
    if not drop_prev_pregnancy:
        return msia_ds, india_ds

    india_df, india_add_on = india_ds[0].copy(), india_ds[1].copy()
    for feature in PREV_PREGNANCY_FEATURES:
        india_df = india_df.drop(columns=[feature], errors="ignore")
        india_add_on = india_add_on.drop(columns=[feature], errors="ignore")
    return msia_ds, [india_df, india_add_on]


def build_variant_folds(variant, seed=SEED):
    """Rebuild every cross-validation fold for one feature variant."""
    drop_prev_pregnancy = variant == "without_prev_preg"
    set_seed(seed)
    msia_ds, india_ds = load_variant_datasets(drop_prev_pregnancy)

    folds = {}
    for prepared in build_harmonized_folds(
        msia_ds,
        india_ds,
        num_of_folds=N_FOLDS_CV,
        label=LABEL,
        accuracy_threshold=ACCURACY_THRESHOLD,
        drop_prev_pregnancy=drop_prev_pregnancy,
    ):
        folds[prepared.fold] = (
            prepared.test_df[prepared.features].reset_index(drop=True),
            prepared.test_Y.to_numpy(dtype=int),
            prepared.country_arr,
            prepared.features,
        )
    return folds


def run():
    """Score every target model per cohort and write the result tables."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for variant in VARIANTS:
        targets = [t for t in TARGETS if t["variant"] == variant]
        if not targets:
            continue
        print(f"\nVARIANT: {variant}  ({len(targets)} model(s))")
        folds = build_variant_folds(variant)

        for target in targets:
            print(f"\n=== {target['name']}  [{target['family']}] ===")
            print(f"    weights dir: {target['path']}/model_weights")

            for fold in range(N_FOLDS_CV):
                test_X, test_Y, country_arr, features = folds[fold]
                model = load_trained_model(
                    target["family"], target["path"], fold,
                    n_features=len(features), dnn_config=target.get("dnn_config"),
                )
                if model is None:
                    print(f"    [fold {fold}] no saved weights found -- skipped "
                          f"(looked in {target['path']}/model_weights/model_{fold}*)")
                    continue

                y_pred, y_prob = predict_labels_and_proba(model, target["family"], test_X)
                split_metrics = evaluate_splits(test_Y, y_pred, y_prob, country_arr)
                append_fold_rows(all_results, split_metrics, target["name"], fold,
                                 extra={"family": target["family"], "variant": variant})
                print(
                    f"    [fold {fold}]  "
                    f"MY(n={int(split_metrics['malaysia']['n_samples'])}) "
                    f"AUC={split_metrics['malaysia']['roc_auc']:.4f} | "
                    f"IN(n={int(split_metrics['india']['n_samples'])}) "
                    f"AUC={split_metrics['india']['roc_auc']:.4f} | "
                    f"ALL AUC={split_metrics['total']['roc_auc']:.4f}"
                )

    if not all_results:
        print("\nNo results produced. Check the weight paths in TARGETS.")
        return None

    return save_country_results(
        all_results, str(SAVE_DIR), prefix="noretrain",
        group_cols=("model", "family", "variant", "eval_split"),
    )


if __name__ == "__main__":
    set_seed(SEED)
    run()
