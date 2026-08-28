"""Domain weighting for the cohort size imbalance (Table 8).

    Run experiment_R0_baseline_retrain.py first (the pretrained CatBoost
    imputers).

Run:
    python -m rebuttals.round1.experiment_R2_6_weighted_training
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sga.config import (
    HARMONIZED_SELECTED_FEATURES,
    LABEL,
    N_FOLDS_CV,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.evaluation.country import evaluate_splits
from sga.models.estimators import train_lr
from sga.pipeline.dataset import (
    load_both_cohorts,
    process_raw_train_and_test_df,
    separate_df_and_df_add_on,
)
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA, prepare_fold

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND1_DIR / "R2_6_weighted_training"

STRATEGIES = {
    "A_equal_country_weight": "equal_country",
    "B_sqrt_ratio_weight": "sqrt_ratio",
    "C_no_weight_baseline": "no_weight",
}
EVAL_SUBSETS = ["combined", "malaysia", "india"]

SUMMARY_METRICS = [
    "balanced_accuracy",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "auprc",
    "sensitivity",
    "specificity",
]


def compute_sample_weights(country_arr, strategy):
    """Per-sample training weights for one weighting strategy."""
    country_arr = np.asarray(country_arr)
    n_malaysia = (country_arr == MALAYSIA).sum()
    n_india = (country_arr == INDIA).sum()
    weights = np.ones(len(country_arr), dtype=np.float64)

    if strategy == "no_weight" or n_india == 0:
        return weights
    if strategy == "equal_country":
        india_weight = n_malaysia / n_india
    elif strategy == "sqrt_ratio":
        india_weight = np.sqrt(n_malaysia / n_india)
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")

    weights[country_arr == INDIA] = india_weight
    return weights


def _training_cohort_sizes(msia_ds, india_ds, fold):
    """Pre-resampling training-row counts per cohort. Consumes no randomness."""
    msia_df, msia_add_on, *_ = separate_df_and_df_add_on(msia_ds, LABEL, id_exists=True)
    india_df, india_add_on, *_ = separate_df_and_df_add_on(india_ds, LABEL, id_exists=False)
    msia_train, _ = process_raw_train_and_test_df(msia_df, msia_add_on, fold, id_exists=True)
    india_train, _ = process_raw_train_and_test_df(
        india_df, india_add_on, fold, id_exists=False)
    return len(msia_train), len(india_train)


def run_experiment():
    """Run the three weighting strategies over the cross-validation folds."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    msia_ds, india_ds = load_both_cohorts()
    print(f"Malaysia: {len(msia_ds[0])} samples, India: {len(india_ds[0])} samples")
    print(f"  [manual imputation] keeping ONLY {sorted(HARMONIZED_SELECTED_FEATURES)}")

    all_results = []
    for strategy_name, strategy_key in STRATEGIES.items():
        print(f"\nStrategy: {strategy_name}")

        for fold in range(N_FOLDS_CV):
            print(f"\n  Fold {fold}/{N_FOLDS_CV - 1}")
            n_msia_train, n_india_train = _training_cohort_sizes(msia_ds, india_ds, fold)
            prepared = prepare_fold(
                msia_ds, india_ds, fold,
                selected_features=HARMONIZED_SELECTED_FEATURES,
            )
            train_X, train_Y = prepared.train_X, prepared.train_Y
            test_X, test_Y = prepared.test_X, prepared.test_Y
            test_country = np.asarray(prepared.country_arr, dtype=int)

            # Weights are defined on the pre-SMOTE rows (Malaysia first, then India);
            # the synthetic rows SMOTENC appends keep weight 1.
            train_country = np.array([MALAYSIA] * n_msia_train + [INDIA] * n_india_train)
            sample_weights = compute_sample_weights(train_country, strategy_key)
            n_synthetic = len(train_X) - len(sample_weights)
            if n_synthetic > 0:
                sample_weights = np.concatenate(
                    [sample_weights, np.ones(n_synthetic, dtype=np.float64)])
            elif n_synthetic < 0:
                sample_weights = sample_weights[: len(train_X)]

            india_mask = train_country == INDIA
            if india_mask.sum() > 0:
                weighted = sample_weights[: len(train_country)]
                print(f"    Train Malaysia: {n_msia_train}, Train India: {n_india_train}")
                print(f"    India weight: {weighted[india_mask][0]:.2f}, "
                      f"Malaysia weight: 1.00")
                print(f"    Effective India total: {weighted[india_mask].sum():.1f}, "
                      f"Effective Malaysia total: {weighted[~india_mask].sum():.1f}")

            print(f"    Train: {len(train_X)} (after SMOTE), Test: {len(test_X)}")
            print(f"    Test Malaysia: {(test_country == MALAYSIA).sum()}, "
                  f"Test India: {(test_country == INDIA).sum()}")

            # Grid search unweighted, then refit the winner with the weights.
            best_model = train_lr(train_X, train_Y)
            best_model.fit(train_X, train_Y, sample_weight=sample_weights)

            y_prob = best_model.predict_proba(test_X)[:, 1]
            y_pred = best_model.predict(test_X).astype(int)
            split_metrics = evaluate_splits(test_Y, y_pred, y_prob, test_country)

            for subset in EVAL_SUBSETS:
                metrics = split_metrics["total" if subset == "combined" else subset]
                row = {"strategy": strategy_name, "fold": fold, "eval_subset": subset}
                row.update(metrics)
                all_results.append(row)

            combined = split_metrics["total"]
            india = split_metrics["india"]
            print(f"    [combined] Bal.Acc={combined['balanced_accuracy']:.4f}  "
                  f"AUC={combined['roc_auc']:.4f}  F1={combined['f1']:.4f}")
            print(f"    [india]    Bal.Acc={india['balanced_accuracy']:.4f}  "
                  f"AUC={india['roc_auc']:.4f}  F1={india['f1']:.4f}")

    results_df = pd.DataFrame(all_results)
    per_fold_path = SAVE_DIR / "per_fold_results.csv"
    results_df.to_csv(per_fold_path, index=False)
    print(f"\nPer-fold results saved to: {per_fold_path}")

    summary_rows = []
    for strategy_name in STRATEGIES:
        for subset in EVAL_SUBSETS:
            mask = ((results_df["strategy"] == strategy_name)
                    & (results_df["eval_subset"] == subset))
            group = results_df[mask]
            if len(group) == 0:
                continue
            row = {"strategy": strategy_name, "eval_subset": subset}
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
    summary_path = SAVE_DIR / "summary_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary results saved to: {summary_path}")

    print("\nWEIGHTED TRAINING EXPERIMENT SUMMARY")
    print(f"Model: Logistic Regression (unified, harmonized), {N_FOLDS_CV}-fold CV")
    for subset in EVAL_SUBSETS:
        print(f"\n  Eval subset: {subset}")
        print(f"  {'Strategy':<30} {'Bal.Acc.':<20} {'ROC AUC':<20} {'F1':<20} "
              f"{'AUPRC':<20} {'Recall':<20}")
        print("  " + "-" * 130)
        for _, row in summary_df[summary_df["eval_subset"] == subset].iterrows():
            print(
                f"  {row['strategy']:<30} "
                f"{row['balanced_accuracy_str']:<20} "
                f"{row['roc_auc_str']:<20} "
                f"{row['f1_str']:<20} "
                f"{row['auprc_str']:<20} "
                f"{row['recall_str']:<20}"
            )

    print("\nKEY COMPARISON: India-only test performance across weighting strategies")
    india_summary = summary_df[summary_df["eval_subset"] == "india"]
    if len(india_summary) > 0:
        columns = ["strategy", "balanced_accuracy_str", "roc_auc_str", "f1_str",
                   "auprc_str"]
        print(india_summary[columns].to_string(index=False))

    print("\nExperiment complete.")
    return results_df, summary_df


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
