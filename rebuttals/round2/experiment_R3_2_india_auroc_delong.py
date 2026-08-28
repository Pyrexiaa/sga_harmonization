"""Is the India AUROC gain statistically distinguishable?

Run:
    python -m rebuttals.round2.experiment_R3_2_india_auroc_delong
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sga.config import (
    HARMONIZED_SELECTED_FEATURES,
    N_FOLDS_CV,
    ROUND2_DIR,
    SEED,
    set_seed,
)
from sga.evaluation.bootstrap import bootstrap_ci
from sga.evaluation.delong import delong_test
from sga.models.estimators import train_lr
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.harmonized_fold import INDIA, prepare_fold

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND2_DIR / "R3_2_india_auroc_delong"


def run_experiment():
    """Compare the baseline and unified arms on the held-out India rows."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    per_fold = []

    for fold in range(N_FOLDS_CV):
        set_seed(SEED)
        msia_ds, india_ds = load_both_cohorts()
        baseline = prepare_fold(
            msia_ds, india_ds, fold, selected_features=(), train_source="india"
        )
        set_seed(SEED)
        msia_ds, india_ds = load_both_cohorts()
        unified = prepare_fold(
            msia_ds,
            india_ds,
            fold,
            selected_features=HARMONIZED_SELECTED_FEATURES,
            train_source="both",
        )
        if baseline["country_arr"] is None or unified["country_arr"] is None:
            print(f"  [warn] fold {fold}: country split unavailable; skip")
            continue

        model_baseline = train_lr(baseline["train_X"], baseline["train_Y"])
        model_unified = train_lr(unified["train_X"], unified["train_Y"])
        prob_baseline = model_baseline.predict_proba(baseline["test_X"])[:, 1]
        prob_unified = model_unified.predict_proba(unified["test_X"])[:, 1]
        india_baseline = baseline["country_arr"] == INDIA
        india_unified = unified["country_arr"] == INDIA
        y_baseline = baseline["test_Y"][india_baseline]
        prob_baseline = prob_baseline[india_baseline]
        y_unified = unified["test_Y"][india_unified]
        prob_unified = prob_unified[india_unified]

        auc_baseline = (
            roc_auc_score(y_baseline, prob_baseline)
            if len(np.unique(y_baseline)) > 1
            else np.nan
        )
        auc_unified = (
            roc_auc_score(y_unified, prob_unified)
            if len(np.unique(y_unified)) > 1
            else np.nan
        )
        row = {
            "fold": fold,
            "n_india": int(india_baseline.sum()),
            "n_india_sga": int(y_baseline.sum()),
            "auc_baseline": auc_baseline,
            "auc_unified": auc_unified,
            "auc_diff": auc_unified - auc_baseline,
        }
        # Paired DeLong only when the India rows align across arms.
        if len(y_baseline) == len(y_unified) and np.array_equal(y_baseline, y_unified):
            _, _, z, p = delong_test(y_unified, prob_unified, prob_baseline)
            row["delong_p"] = p
            row["delong_z"] = z
        else:
            row["delong_p"] = np.nan
            row["delong_z"] = np.nan
            row["note"] = "India rows differ across arms; use bootstrap CI of diff"
        per_fold.append(row)

    per_fold_df = pd.DataFrame(per_fold)
    per_fold_df.to_csv(SAVE_DIR / "india_delong_per_fold.csv", index=False)

    differences = per_fold_df["auc_diff"].dropna().tolist()
    mean_diff, lo, hi = bootstrap_ci(differences)
    summary = {
        "india_sga_events_total": int(per_fold_df["n_india_sga"].sum()),
        "mean_auc_baseline": float(per_fold_df["auc_baseline"].mean()),
        "mean_auc_unified": float(per_fold_df["auc_unified"].mean()),
        "mean_auc_diff": mean_diff,
        "diff_ci_low": lo,
        "diff_ci_high": hi,
        "diff_ci_excludes_zero": bool((lo > 0) or (hi < 0)),
        "median_delong_p": float(per_fold_df["delong_p"].median()),
    }
    pd.Series(summary).to_csv(SAVE_DIR / "india_delong_summary.csv")
    print(per_fold_df.to_string(index=False))
    print("\nSUMMARY:", summary)
    print(
        "\nInterpretation: if diff_ci_excludes_zero is False and median_delong_p > 0.05,"
        " the India improvement (0.5605 -> 0.6753, the ~11% headline) is NOT"
        " statistically distinguishable and must be reported as such."
    )
    print(f"Saved to: {SAVE_DIR}")
    return per_fold_df, summary


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
