"""Internal cross-validation versus external test (appendix Table S3).

Backs the Results section "Model Internal Validation Performances" and appendix
Table S3. For each of the six classifiers and each development fold 0-3 it scores
the SAME fold-model twice:

* **Validation** -- on that fold's own held-out development partition
  (``build_development_folds``), i.e. the internal cross-validation estimate.
* **Test** -- on the held-out external partition, fold 4
  (``build_external_folds``), the same rows every other reported result uses.

Reporting both from one script is what makes the optimism claim checkable: the
manuscript quotes a mean internal-validation AUROC of 0.803 against a test AUROC
of 0.811 for Logistic Regression, and a maximum internal-to-test gap of 0.017 for
five of six classifiers with CatBoost the exception at nearly 0.055.

Weights come from the harmonized (``ute_ari`` + ``af``) arm, so the numbers are
the manuscript's model rather than the full-imputation sweep.

Prerequisites:
    python -m rebuttals.round1.experiment_R0_baseline_retrain
    python -m rebuttals.round1.experiment_R0_baseline_retrain_manual

Run:
    python -m rebuttals.round1.experiment_R2_10_internal_vs_external
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sga.config import (
    EXTERNAL_TEST_FOLD,
    HARMONIZED_SELECTED_FEATURES,
    MODEL_DISPLAY_NAMES,
    N_FOLDS_CV,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.evaluation.metrics import full_metrics
from sga.pipeline.model_io import load_trained_model, predict_labels_and_proba

from rebuttals.round1.experiment_R0_baseline_retrain import (
    DNN_CONFIGS,
    HARMONIZED_WEIGHTS_DIR,
    build_development_folds,
    build_external_folds,
)

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND1_DIR / "R2_10_internal_vs_external"

#: Which trained DNN configuration represents the "Neural Network" row.
DNN_CONFIG_INDEX = 0

#: Display name -> (family, weights sub-directory, DNN config index).
MODEL_LOADERS = {
    MODEL_DISPLAY_NAMES["lr"]: ("ml", f"generalized_lr_{SEED}", None),
    MODEL_DISPLAY_NAMES["dnn"]: (
        "dnn",
        f"generalized_dnn_{SEED}_{DNN_CONFIG_INDEX}",
        DNN_CONFIG_INDEX,
    ),
    MODEL_DISPLAY_NAMES["rf"]: ("ml", f"generalized_rf_{SEED}", None),
    MODEL_DISPLAY_NAMES["svc"]: ("ml", f"generalized_svc_{SEED}", None),
    MODEL_DISPLAY_NAMES["catboost"]: ("catboost", f"generalized_catboost_{SEED}", None),
    MODEL_DISPLAY_NAMES["stacking"]: ("ml", f"generalized_stacking_{SEED}", None),
}

#: The five metrics tabulated in Table S3, in the table's row order.
TABLE_S3_METRICS = [
    ("roc_auc", "AUROC"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("f1", "F1"),
    ("precision", "Precision"),
    ("recall", "Recall"),
]


def _score(display_name, fold, X, y_true, n_features):
    """Metric row for one fold-model on one evaluation partition, or None."""
    family, subdir, dnn_index = MODEL_LOADERS[display_name]
    dnn_config = DNN_CONFIGS[dnn_index] if dnn_index is not None else None
    try:
        model = load_trained_model(
            family,
            str(HARMONIZED_WEIGHTS_DIR / subdir / "malaysia_tri3"),
            fold,
            n_features=n_features,
            dnn_config=dnn_config,
        )
    except Exception as error:  # noqa: BLE001 - a missing arm must not stop the sweep
        print(f"    [skip] {display_name} fold {fold}: {error}")
        return None
    if model is None:
        print(f"    [skip] {display_name}: no weights for fold {fold}")
        return None

    y_pred, y_prob = predict_labels_and_proba(model, family, X)
    return full_metrics(np.asarray(y_true), y_pred, y_prob)


def run_experiment():
    """Score every fold-model on its own validation fold and on fold 4."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    development = build_development_folds(selected_features=HARMONIZED_SELECTED_FEATURES)
    external = build_external_folds(selected_features=HARMONIZED_SELECTED_FEATURES)

    rows = []
    for display_name in MODEL_LOADERS:
        print(f"\n{display_name}")
        for fold in range(N_FOLDS_CV):
            if fold in development:
                X, y_true, _, features = development[fold]
                metrics = _score(display_name, fold, X, y_true, len(features))
                if metrics is not None:
                    rows.append({"model": display_name, "set": "Validation",
                                 "fold": fold, **metrics})
            if fold in external:
                _, X, y_true, _, features = external[fold]
                metrics = _score(display_name, fold, X, y_true, len(features))
                if metrics is not None:
                    rows.append({"model": display_name, "set": "Test",
                                 "fold": fold, **metrics})

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        print("No weights found -- run the two R0 prerequisites first.")
        return per_fold
    per_fold.to_csv(SAVE_DIR / "internal_vs_external_per_fold.csv", index=False)

    # Table S3 layout: model x metric x set, folds across columns, then the mean.
    table_rows = []
    for display_name in MODEL_LOADERS:
        for column, label in TABLE_S3_METRICS:
            for set_name in ("Validation", "Test"):
                subset = per_fold[
                    (per_fold["model"] == display_name) & (per_fold["set"] == set_name)
                ].set_index("fold")
                if subset.empty:
                    continue
                row = {"Model": display_name, "Metric": label, "Set": set_name}
                for fold in range(N_FOLDS_CV):
                    value = subset[column].get(fold, np.nan)
                    row[f"Fold {fold}"] = round(float(value), 3) if pd.notna(value) else np.nan
                row["Mean"] = round(float(subset[column].mean()), 3)
                table_rows.append(row)

    table = pd.DataFrame(table_rows)
    table.to_csv(SAVE_DIR / "tableS3_internal_vs_external.csv", index=False)

    # The optimism statement in the Results text: mean validation vs mean test AUROC.
    optimism = []
    for display_name in MODEL_LOADERS:
        auroc = table[(table["Model"] == display_name) & (table["Metric"] == "AUROC")]
        validation = auroc[auroc["Set"] == "Validation"]["Mean"]
        test = auroc[auroc["Set"] == "Test"]["Mean"]
        if validation.empty or test.empty:
            continue
        optimism.append({
            "model": display_name,
            "mean_validation_auroc": float(validation.iloc[0]),
            "mean_test_auroc": float(test.iloc[0]),
            "validation_minus_test": round(float(validation.iloc[0] - test.iloc[0]), 4),
        })
    optimism_df = pd.DataFrame(optimism)
    optimism_df.to_csv(SAVE_DIR / "optimism_summary.csv", index=False)

    print(f"\nTable S3 (external fold = {EXTERNAL_TEST_FOLD})")
    print(table.to_string(index=False))
    print("\nInternal validation minus external test AUROC")
    print(optimism_df.to_string(index=False))
    print(f"\nSaved to: {SAVE_DIR}")
    return table


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
