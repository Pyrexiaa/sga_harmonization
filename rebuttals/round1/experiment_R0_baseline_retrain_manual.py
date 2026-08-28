"""Round-1 prerequisite: unified models for MANUAL imputation-feature combinations.

    Run experiment_R0_baseline_retrain.py first (the pretrained CatBoost
    imputers).

Run:
    python -m rebuttals.round1.experiment_R0_baseline_retrain_manual
"""

from __future__ import annotations

import numpy as np

from sga.config import (
    ACCURACY_THRESHOLD,
    EXTERNAL_TEST_FOLD,
    N_FOLDS_CV,
    SEED,
    set_seed,
)
from sga.evaluation.country import append_fold_rows, evaluate_splits, save_country_results
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.model_io import (
    load_trained_model,
    predict_labels_and_proba,
    save_predictions,
    weights_exist,
)
from sga.pipeline.train_unified import (
    train_catboost_unified,
    train_dnn_unified,
    train_sklearn_unified,
)

from rebuttals.round1.experiment_R0_baseline_retrain import (
    DNN_CONFIGS,
    ML_MODELS,
    SELECTED_IMPUTATION_DIR,
    build_external_folds,
)

FORCE_RETRAIN = False

# Combination name -> the cross-domain features to impute (common features are always
# kept).
SELECTED_COMBINATIONS = {
    "sel_ute_ari": ["ute_ari"],
    "sel_ute_ari_af": ["ute_ari", "af"],
    "sel_ute_ari_af_last_preg_fgr": ["ute_ari", "af", "last_preg_fgr"],
    "sel_ute_ari_af_last_preg_normal": ["ute_ari", "af", "last_preg_normal"],
    "sel_ute_ari_af_last_preg_fgr_last_preg_normal":
        ["ute_ari", "af", "last_preg_fgr", "last_preg_normal"],
}


def _combination_models(combination_name):
    """Model name -> (family, results root, DNN config index) for one combination."""
    base = SELECTED_IMPUTATION_DIR / combination_name
    models = {
        "catboost": ("catboost", base / f"generalized_catboost_{SEED}", None),
    }
    for model_type in ML_MODELS:
        models[model_type] = ("ml", base / f"generalized_{model_type}_{SEED}", None)
    for index in range(len(DNN_CONFIGS)):
        models[f"dnn_cfg{index}"] = (
            "dnn", base / f"generalized_dnn_{SEED}_{index}", index)
    return models


def train_combination(combination_name, selected_features, smoting=True):
    """Train every classifier family for one manual feature combination."""
    print(f"\nTRAIN selected-imputation baselines -- {combination_name}: "
          f"{selected_features}")
    base = SELECTED_IMPUTATION_DIR / combination_name
    common = {
        "smoting": smoting,
        "undersampling": False,
        "accuracy_threshold": ACCURACY_THRESHOLD,
        "selected_features": selected_features,
    }

    save_dir = base / f"generalized_catboost_{SEED}" / "malaysia_tri3"
    if FORCE_RETRAIN or not weights_exist(str(save_dir), N_FOLDS_CV):
        msia_ds, india_ds = load_both_cohorts()
        train_catboost_unified(msia_ds, india_ds, download_path=str(save_dir), **common)
    else:
        print(f"  [skip] CatBoost ({combination_name}) already trained")

    for model_type in ML_MODELS:
        save_dir = base / f"generalized_{model_type}_{SEED}" / "malaysia_tri3"
        if FORCE_RETRAIN or not weights_exist(str(save_dir), N_FOLDS_CV):
            msia_ds, india_ds = load_both_cohorts()
            train_sklearn_unified(msia_ds, india_ds, download_path=str(save_dir),
                                  model_type=model_type, **common)
        else:
            print(f"  [skip] ML {model_type} ({combination_name}) already trained")

    for index, config in enumerate(DNN_CONFIGS):
        save_dir = base / f"generalized_dnn_{SEED}_{index}" / "malaysia_tri3"
        if FORCE_RETRAIN or not weights_exist(str(save_dir), N_FOLDS_CV):
            msia_ds, india_ds = load_both_cohorts()
            train_dnn_unified(msia_ds, india_ds, download_path=str(save_dir),
                              hyperparameters=config, model_size="large", **common)
        else:
            print(f"  [skip] DNN config {index} ({combination_name}) already trained")


def external_inference_combination(combination_name, selected_features):
    """Score one combination's saved fold-models on the external fold."""
    print(f"\nEXTERNAL TEST (fold {EXTERNAL_TEST_FOLD}) -- {combination_name}")
    base = SELECTED_IMPUTATION_DIR / combination_name
    output_root = base / "external_test_predictions"
    harmonized = build_external_folds(selected_features=selected_features)

    metric_rows = []
    for name, (family, results_root, dnn_index) in _combination_models(
            combination_name).items():
        dnn_config = DNN_CONFIGS[dnn_index] if dnn_index is not None else None
        print(f"\n--- {name} ---")
        probability_stack, raw_df, y_true, country_arr = [], None, None, None
        for fold in range(N_FOLDS_CV):
            raw_df, test_X, y_true, country_arr, features = harmonized[fold]
            model = load_trained_model(
                family, str(results_root / "malaysia_tri3"), fold,
                n_features=len(features), dnn_config=dnn_config,
            )
            if model is None:
                print(f"    [fold {fold}] weights missing -- skipped")
                continue
            y_pred, y_prob = predict_labels_and_proba(model, family, test_X)
            save_predictions(str(output_root / name / f"fold_{fold}.csv"),
                             raw_df, y_true, y_pred, y_prob, country_arr)
            append_fold_rows(
                metric_rows, evaluate_splits(y_true, y_pred, y_prob, country_arr),
                name, fold, extra={"combo": combination_name},
            )
            probability_stack.append(y_prob)

        if probability_stack and raw_df is not None:
            ensemble = np.mean(np.vstack(probability_stack), axis=0)
            ensemble_pred = (ensemble >= 0.5).astype(int)
            save_predictions(str(output_root / name / "ensemble.csv"),
                             raw_df, y_true, ensemble_pred, ensemble, country_arr)
            append_fold_rows(
                metric_rows,
                evaluate_splits(y_true, ensemble_pred, ensemble, country_arr),
                name, "ensemble", extra={"combo": combination_name},
            )

    if metric_rows:
        save_country_results(metric_rows, str(base), prefix="external_metrics",
                             group_cols=("model", "combo", "eval_split"))
    print(f"\n  External predictions saved under: {output_root}")
    return metric_rows


def run_experiment():
    """Train and externally test every manual imputation-feature combination."""
    for combination_name, selected_features in SELECTED_COMBINATIONS.items():
        train_combination(combination_name, selected_features)
        external_inference_combination(combination_name, selected_features)
    print("\nAll selected-imputation baselines trained and externally tested.")


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
