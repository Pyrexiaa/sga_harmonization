"""Unified results reported split by cohort.

    Run experiment_R0_baseline_retrain.py first (the pretrained CatBoost
    imputers).

Run:
    python -m rebuttals.round1.experiment_R2_7_split_unified_by_country
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sga.config import ACCURACY_THRESHOLD, N_FOLDS_CV, ROUND1_DIR, SEED, set_seed
from sga.evaluation.country import evaluate_splits
from sga.evaluation.metrics import EVAL_SPLITS
from sga.models.estimators import train_lr
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.train_unified import build_harmonized_folds

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND1_DIR / "R2_7_split_unified_by_country"

SUMMARY_METRICS = [
    "balanced_accuracy", "roc_auc", "f1", "precision", "recall",
    "sensitivity", "specificity", "ppv", "npv", "auprc", "brier_score",
]


def run_experiment():
    """Train the unified Logistic Regression and report it per cohort."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    msia_ds, india_ds = load_both_cohorts()
    print(f"Malaysia: {len(msia_ds[0])} samples (after removing fold 4)")
    print(f"India:    {len(india_ds[0])} samples (after removing fold 4)")

    all_results = []
    for prepared in build_harmonized_folds(
        msia_ds, india_ds, num_of_folds=N_FOLDS_CV,
        accuracy_threshold=ACCURACY_THRESHOLD,
    ):
        fold = prepared.fold
        country_arr = np.asarray(prepared.country_arr, dtype=int)
        n_malaysia = int((country_arr == 0).sum())
        n_india = int((country_arr == 1).sum())
        print(f"\nFold {fold}/{N_FOLDS_CV - 1}")
        print(f"  Malaysia test: {n_malaysia}, India test: {n_india}")
        print(f"  Train: {len(prepared.train_X)} (after SMOTE)")
        print(f"  Test:  {len(prepared.test_X)} "
              f"(Malaysia: {n_malaysia}, India: {n_india})")

        print("  Training Logistic Regression (GridSearchCV)...")
        model = train_lr(prepared.train_X, prepared.train_Y)
        y_pred = model.predict(prepared.test_X).astype(int)
        y_prob = model.predict_proba(prepared.test_X)[:, 1]
        y_true = np.asarray(prepared.test_Y)

        split_metrics = evaluate_splits(y_true, y_pred, y_prob, country_arr)
        for split in EVAL_SPLITS:
            row = {"model": "LR", "fold": fold, "eval_split": split}
            row.update(split_metrics[split])
            all_results.append(row)
            metrics = split_metrics[split]
            print(f"    [{split}] Bal.Acc={metrics['balanced_accuracy']:.4f}  "
                  f"AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")

        predictions = prepared.test_df[prepared.features].copy()
        predictions["actual"] = y_true
        predictions["country"] = country_arr
        predictions["LR_predicted"] = y_pred
        predictions["LR_probability"] = y_prob
        predictions_path = SAVE_DIR / f"test_predictions_fold_{fold}.csv"
        predictions.to_csv(predictions_path, index=False)
        print(f"  Saved test predictions to: {predictions_path}")

    results_df = pd.DataFrame(all_results)
    per_fold_path = SAVE_DIR / "per_fold_results.csv"
    results_df.to_csv(per_fold_path, index=False)
    print(f"\nPer-fold results saved to: {per_fold_path}")

    summary_rows = []
    for split in EVAL_SPLITS:
        group = results_df[(results_df["model"] == "LR")
                           & (results_df["eval_split"] == split)]
        if len(group) == 0:
            continue
        row = {"model": "LR", "eval_split": split}
        for column in SUMMARY_METRICS:
            values = group[column].dropna()
            if len(values) > 0:
                row[f"{column}_mean"] = values.mean()
                row[f"{column}_std"] = values.std()
                row[f"{column}_str"] = f"{values.mean():.4f} +/- {values.std():.4f}"
            else:
                row[f"{column}_mean"] = float("nan")
                row[f"{column}_std"] = float("nan")
                row[f"{column}_str"] = "N/A"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = SAVE_DIR / "summary_by_country.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary results saved to: {summary_path}")

    print("\nSPLIT UNIFIED RESULTS BY COUNTRY")
    print("\n--- Model: Logistic Regression (unified pipeline) ---")
    print(f"  {'Split':<12} {'Bal.Acc.':<22} {'ROC AUC':<22} {'F1':<22} "
          f"{'AUPRC':<22} {'Brier':<22} {'Sens.':<22} {'Spec.':<22}")
    print("  " + "-" * 156)
    for _, row in summary_df.iterrows():
        print(
            f"  {row['eval_split']:<12} "
            f"{row['balanced_accuracy_str']:<22} "
            f"{row['roc_auc_str']:<22} "
            f"{row['f1_str']:<22} "
            f"{row['auprc_str']:<22} "
            f"{row['brier_score_str']:<22} "
            f"{row['sensitivity_str']:<22} "
            f"{row['specificity_str']:<22}"
        )

    print("\nKEY COMPARISON: Malaysia vs India test performance")
    malaysia_row = summary_df[summary_df["eval_split"] == "malaysia"]
    india_row = summary_df[summary_df["eval_split"] == "india"]
    if len(malaysia_row) > 0 and len(india_row) > 0:
        malaysia_auc = malaysia_row.iloc[0]["roc_auc_mean"]
        india_auc = india_row.iloc[0]["roc_auc_mean"]
        malaysia_bacc = malaysia_row.iloc[0]["balanced_accuracy_mean"]
        india_bacc = india_row.iloc[0]["balanced_accuracy_mean"]
        print(f"  LR  AUC: Malaysia={malaysia_auc:.4f}, India={india_auc:.4f}, "
              f"Delta={malaysia_auc - india_auc:+.4f}  |  "
              f"Bal.Acc: Malaysia={malaysia_bacc:.4f}, India={india_bacc:.4f}, "
              f"Delta={malaysia_bacc - india_bacc:+.4f}")

    print("\nExperiment complete.")
    return results_df, summary_df


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
