"""Malaysia vs India disparity of the pretrained models.

SUPPORTING ANALYSIS, not the source of the manuscript's numbers. The
"Cohort-Level Fairness Analysis" section and appendix Table S5 are produced by
``rebuttals.round2.experiment_R2_5_fairness_uncertainty``, which differs from this
script in three ways that change the values:

* it uses the Platt-CALIBRATED probabilities, whereas this script thresholds the
  raw model output at 0.5 (the Platt map is monotone, so the two select different
  patients at the same nominal cut-off);
* it reports SIGNED differences with Wilson and Newcombe score intervals, whereas
  this script reports absolute gaps with no uncertainty at all;
* it is computed once on the external test fold, whereas this script reports the
  development folds and the external fold under the pretrained fold-models.

What this script is still for: checking that the direction of the disparity is
stable across development folds and across the harmonized and common-feature
models, without retraining anything.

    Run experiment_R0_baseline_retrain.py, then
    experiment_R0_baseline_retrain_manual.py, and
    experiment_R2_5_imputation_ablation.py for the common-features baseline.

Run:
    python -m rebuttals.round1.experiment_R2_8_fairness_metrics
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from sga.config import (
    EXTERNAL_TEST_FOLD,
    HARMONIZED_SELECTED_FEATURES,
    N_FOLDS_CV,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.evaluation.fairness import compute_fairness_metrics, compute_group_rates
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA, prepare_fold
from sga.pipeline.model_io import load_trained_model, predict_labels_and_proba

from rebuttals.round1.experiment_R0_baseline_retrain import (
    HARMONIZED_WEIGHTS_DIR,
    build_external_folds,
)
from rebuttals.round1.experiment_R2_5_imputation_ablation import (
    WEIGHTS_DIR as ABLATION_WEIGHTS_DIR,
)

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND1_DIR / "R2_8_fairness_metrics"
LR_WEIGHTS_DIR = HARMONIZED_WEIGHTS_DIR / f"generalized_lr_{SEED}" / "malaysia_tri3"
COMMON_LR_WEIGHTS_DIR = ABLATION_WEIGHTS_DIR / "A_no_imputation"

GAP_METRICS = ["eod", "dpd", "equalized_odds_diff", "predictive_parity"]
RATE_METRICS = [
    "tpr_malaysia", "tpr_india", "fpr_malaysia", "fpr_india",
    "ppv_malaysia", "ppv_india", "pred_rate_malaysia", "pred_rate_india",
]


def fairness_row(y_true, y_pred, country_arr):
    """Cohort gap metrics plus the per-cohort rates behind them."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    country_arr = np.asarray(country_arr)

    row = dict(compute_fairness_metrics(y_true, y_pred, country_arr))
    # Provenance marker: these gaps are absolute, uncalibrated and interval-free.
    row["probabilities"] = "uncalibrated"
    row["gap_sign"] = "absolute"
    for name, code in (("malaysia", MALAYSIA), ("india", INDIA)):
        mask = country_arr == code
        rates = compute_group_rates(y_true, y_pred, mask)
        row[f"ppv_{name}"] = rates["ppv"]
        row[f"pred_rate_{name}"] = rates["pred_rate"]
        row[f"n_{name}"] = rates["n"]
        row[f"n_positive_{name}"] = int(y_true[mask].sum()) if mask.sum() else 0
    return row


def _summary_table(frame, columns):
    """Long-format mean / std / formatted summary over the requested columns."""
    return pd.DataFrame({
        "metric": columns,
        "mean": [frame[c].mean() for c in columns],
        "std": [frame[c].std() for c in columns],
        "str": [f"{frame[c].mean():.4f} +/- {frame[c].std():.4f}" for c in columns],
    })


def _print_fairness(prefix, fairness):
    """Console line for one fold's fairness metrics."""
    print(f"  {prefix}EOD={fairness['eod']:.4f}  DPD={fairness['dpd']:.4f}  "
          f"EqOdds={fairness['equalized_odds_diff']:.4f}  "
          f"PredParity={fairness['predictive_parity']:.4f}")


def run_experiment():
    """Per-fold cross-validation fairness of the pretrained harmonized LR."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    msia_ds, india_ds = load_both_cohorts()
    print(f"Malaysia: {len(msia_ds[0])} samples (after removing fold 4)")
    print(f"India:    {len(india_ds[0])} samples (after removing fold 4)")
    print(f"  [manual imputation] keeping ONLY {sorted(HARMONIZED_SELECTED_FEATURES)}")

    all_rows = []
    for fold in range(N_FOLDS_CV):
        print(f"\nFold {fold}/{N_FOLDS_CV - 1}")
        prepared = prepare_fold(
            msia_ds, india_ds, fold, selected_features=HARMONIZED_SELECTED_FEATURES)
        country_arr = np.asarray(prepared.country_arr, dtype=int)
        print(f"  Train: {len(prepared.train_X)} (after SMOTE)")
        print(f"  Test:  {len(prepared.test_X)} "
              f"(Malaysia: {int((country_arr == MALAYSIA).sum())}, "
              f"India: {int((country_arr == INDIA).sum())})")

        print("  Loading pretrained Logistic Regression weights (no retraining)...")
        model = load_trained_model("ml", str(LR_WEIGHTS_DIR), fold)
        if model is None:
            print(f"  [fold {fold}] LR weights missing under {LR_WEIGHTS_DIR} -- skipped")
            continue
        y_pred, y_prob = predict_labels_and_proba(model, "ml", prepared.test_X)
        y_true = np.asarray(prepared.test_Y)

        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"  Overall: Bal.Acc={balanced_accuracy:.4f}  AUC={auc:.4f}  F1={f1:.4f}")

        fairness = fairness_row(y_true, y_pred, country_arr)
        row = {"fold": fold, "balanced_accuracy": balanced_accuracy,
               "roc_auc": auc, "f1": f1}
        row.update(fairness)
        all_rows.append(row)

        _print_fairness("", fairness)
        print(f"  TPR: Malaysia={fairness['tpr_malaysia']:.4f}, "
              f"India={fairness['tpr_india']:.4f}")
        print(f"  FPR: Malaysia={fairness['fpr_malaysia']:.4f}, "
              f"India={fairness['fpr_india']:.4f}")
        print(f"  PPV: Malaysia={fairness['ppv_malaysia']:.4f}, "
              f"India={fairness['ppv_india']:.4f}")

    if not all_rows:
        print("\nNo LR weights found -- cross-validation fairness skipped.")
        return None

    results_df = pd.DataFrame(all_rows)
    per_fold_path = SAVE_DIR / "fairness_per_fold.csv"
    results_df.to_csv(per_fold_path, index=False)
    print(f"\nPer-fold results saved to: {per_fold_path}")

    summary_df = _summary_table(
        results_df, GAP_METRICS + RATE_METRICS + ["balanced_accuracy", "roc_auc", "f1"])
    summary_path = SAVE_DIR / "fairness_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    print("\nFAIRNESS METRICS SUMMARY")
    print("Logistic Regression, unified harmonized pipeline, Malaysia vs India")
    print("Lower values indicate a smaller cohort gap (0 = no gap)")
    print(f"\n  {'Metric':<30} {'Value (mean +/- std)':<30}")
    print(f"  {'-' * 60}")
    for _, row in summary_df.iterrows():
        print(f"  {row['metric']:<30} {row['str']:<30}")

    print("\nNOTE: the manuscript's cohort-level fairness numbers, their Wilson and")
    print("  Newcombe intervals and appendix Table S5 come from")
    print("  rebuttals.round2.experiment_R2_5_fairness_uncertainty, on the")
    print("  CALIBRATED probabilities of the external test fold. The values below")
    print("  are the uncalibrated cross-validation view and will not match.")
    print("\nINTERPRETATION GUIDE")
    print("  Reported as a SITE/COHORT PERFORMANCE-DISPARITY analysis, not a")
    print("  comprehensive fairness assessment: no individual-level demographic")
    print("  or comorbidity variables are available.")
    print("  EOD (Equal Opportunity Difference): |TPR_Malaysia - TPR_India|")
    print("    Whether SGA is detected equally well in both cohorts. No")
    print("    pre-specified acceptability threshold exists, so no gap is")
    print("    described as 'acceptable'; report the DIRECTION of the gap and")
    print("    its clinical consequence. A lower Indian TPR means SGA cases in")
    print("    the under-represented cohort are more likely to be missed.")
    print("  DPD (Demographic Parity Difference): |P(pred=1|MY) - P(pred=1|IN)|")
    print("  Equalized Odds Difference: max(|TPR gap|, |FPR gap|) -- strictest.")
    print("  Predictive Parity: |PPV_Malaysia - PPV_India|.")
    return results_df, summary_df


def run_external_fairness():
    """Fairness of the harmonized LR on the held-out external fold."""
    print(f"\nORIGINAL LR fairness on EXTERNAL fold-{EXTERNAL_TEST_FOLD} test (no retraining)")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    harmonized = build_external_folds(selected_features=HARMONIZED_SELECTED_FEATURES)
    rows = []
    for fold in range(N_FOLDS_CV):
        _, test_X, y_true, country_arr, _ = harmonized[fold]
        model = load_trained_model("ml", str(LR_WEIGHTS_DIR), fold)
        if model is None:
            print(f"  [fold {fold}] LR weights missing -- skipped")
            continue
        y_pred, _ = predict_labels_and_proba(model, "ml", test_X)
        fairness = fairness_row(y_true, y_pred, country_arr)
        rows.append({"model": "LR_generalized", "fold": fold, **fairness})
        _print_fairness(f"[fold {fold} model on fold-{EXTERNAL_TEST_FOLD}] ", fairness)

    if not rows:
        print("  no LR weights found -- external fairness skipped")
        return None

    frame = pd.DataFrame(rows)
    per_fold_path = SAVE_DIR / "fairness_external_fold4_per_fold.csv"
    frame.to_csv(per_fold_path, index=False)
    _summary_table(frame, GAP_METRICS).to_csv(
        SAVE_DIR / "fairness_external_fold4_summary.csv", index=False)
    print(f"  saved {per_fold_path}")
    return frame


def run_common_lr_fairness():
    """Fairness of the common-features LR baseline, cross-validated and external."""
    print("\nCOMMON-FEATURES LR fairness (loaded weights, no retraining)")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    msia_ds, india_ds = load_both_cohorts()
    external = build_external_folds(selected_features=[])

    rows = []
    for fold in range(N_FOLDS_CV):
        model = load_trained_model(
            "ml", str(COMMON_LR_WEIGHTS_DIR), fold, weights_subdir="")
        if model is None:
            print(f"  [fold {fold}] common-LR weights missing under "
                  f"{COMMON_LR_WEIGHTS_DIR} -- skipped")
            continue

        prepared = prepare_fold(msia_ds, india_ds, fold, selected_features=())
        cv_pred, _ = predict_labels_and_proba(model, "ml", prepared.test_X)
        rows.append({
            "model": "LR_common", "regime": "cv", "fold": fold,
            **fairness_row(prepared.test_Y, cv_pred, prepared.country_arr),
        })

        _, test_X, y_true, country_arr, _ = external[fold]
        external_pred, _ = predict_labels_and_proba(model, "ml", test_X)
        rows.append({
            "model": "LR_common", "regime": f"external_fold{EXTERNAL_TEST_FOLD}",
            "fold": fold, **fairness_row(y_true, external_pred, country_arr),
        })

    if not rows:
        print("  no common-LR weights found -- skipped")
        return None

    frame = pd.DataFrame(rows)
    per_fold_path = SAVE_DIR / "fairness_common_lr_per_fold.csv"
    frame.to_csv(per_fold_path, index=False)
    frame.groupby("regime")[GAP_METRICS].agg(["mean", "std"]).to_csv(
        SAVE_DIR / "fairness_common_lr_summary.csv")
    print(f"  saved {per_fold_path}")
    return frame


def run():
    """Run all three fairness regimes."""
    run_experiment()
    run_external_fairness()
    run_common_lr_fairness()


if __name__ == "__main__":
    set_seed(SEED)
    run()
