"""Held-out external-fold inference for the unified models."""

from __future__ import annotations

from sga.config import EXTERNAL_TEST_FOLD, LABEL, N_FOLDS_CV, SEED, set_seed
from sga.evaluation.country import append_fold_rows, evaluate_splits, save_country_results
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.model_io import (  # noqa: F401 - CATBOOST/SKLEARN/DNN re-exported
    CATBOOST,
    DNN,
    SKLEARN,
    load_trained_model,
    predict_labels_and_proba,
    weights_exist,
)
from sga.pipeline.train_unified import build_harmonized_folds

# Backwards-compatible names for the single implementation in `model_io`.
load_model = load_trained_model
predict = predict_labels_and_proba
weights_present = weights_exist


def run_external_inference(
    family,
    model_name,
    download_path,
    num_of_folds=N_FOLDS_CV,
    label=LABEL,
    accuracy_threshold=0.8,
    drop_prev_pregnancy=False,
    train_source=None,
    smoting=True,
    with_validation=False,
    dnn_config=None,
    model_size="large",
    external_test_fold=EXTERNAL_TEST_FOLD,
    seed=SEED,
    chart=None,
    selected_features=None,
    feature_subset=None,
):
    """Score every saved fold-model on the held-out external fold."""
    if not weights_present(download_path, num_of_folds):
        raise FileNotFoundError(
            f"No complete set of fold weights under {download_path}/model_weights; "
            "run the matching 03* training script first."
        )

    # One seeding before a continuous pass over the folds:
    set_seed(seed)
    load_kwargs = {"chart": chart} if chart else {}
    msia_ds, india_ds = load_both_cohorts(exclude_external_fold=True, **load_kwargs)
    msia_full, india_full = load_both_cohorts(exclude_external_fold=False, **load_kwargs)

    rows = []
    for prepared in build_harmonized_folds(
        msia_ds,
        india_ds,
        num_of_folds=num_of_folds,
        label=label,
        accuracy_threshold=accuracy_threshold,
        drop_prev_pregnancy=drop_prev_pregnancy,
        train_source=train_source,
        smoting=smoting,
        with_validation=with_validation,
        msia_ds_full=msia_full,
        india_ds_full=india_full,
        external_test_fold=external_test_fold,
        selected_features=selected_features,
        feature_subset=feature_subset,
        imputer_seed=seed,
    ):
        fold = prepared.fold
        model = load_model(
            family,
            download_path,
            fold,
            n_features=len(prepared.features),
            dnn_config=dnn_config,
            model_size=model_size,
        )
        if model is None:
            print(f"  [fold {fold}] weights missing, skipped")
            continue

        test_X = prepared.test_df[prepared.features]
        test_Y = prepared.test_df[label].values.astype(int)
        y_pred, y_prob = predict(model, family, test_X)

        split_metrics = evaluate_splits(test_Y, y_pred, y_prob, prepared.country_arr)
        append_fold_rows(
            rows, split_metrics, model_name, fold, extra={"eval_fold": external_test_fold}
        )
        print(
            f"  [fold-{fold} model -> fold-{external_test_fold} external] "
            f"MY AUC={split_metrics['malaysia']['roc_auc']:.4f} | "
            f"IN AUC={split_metrics['india']['roc_auc']:.4f} | "
            f"ALL AUC={split_metrics['total']['roc_auc']:.4f}"
        )

    if rows:
        save_country_results(
            rows,
            download_path,
            prefix=f"external_fold{external_test_fold}",
            model_name=model_name,
        )
    else:
        print(f"No fold-models could be scored under {download_path}")
    return rows
