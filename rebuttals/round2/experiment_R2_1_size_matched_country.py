"""Size-matched country-specific comparison.

Run:
    python -m rebuttals.round2.experiment_R2_1_size_matched_country
"""

from __future__ import annotations

import warnings

import pandas as pd

from sga.config import (
    HARMONIZED_SELECTED_FEATURES,
    N_FOLDS_CV,
    ROUND2_DIR,
    SEED,
    set_seed,
)
from sga.evaluation.bootstrap import bootstrap_ci
from sga.evaluation.delong import delong_test
from sga.evaluation.metrics import basic_metrics
from sga.models.estimators import train_lr
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.harmonized_fold import INDIA, prepare_fold

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND2_DIR / "R2_1_size_matched_country"

#: Subsample seeds per fold; the CIs are taken across folds x repeats.
N_REPEATS = 10

ARMS = [
    ("india_only", {"selected": (), "train_source": "india"}),
    ("malaysia_only", {"selected": (), "train_source": "malaysia"}),
    ("pooled_common", {"selected": (), "train_source": "both"}),
    ("pooled_harmonized", {"selected": HARMONIZED_SELECTED_FEATURES, "train_source": "both"}),
]


def india_only_train_size(msia_ds, india_ds, fold):
    """Return the matched training size: the pre-SMOTE India-only row count."""
    fold_data = prepare_fold(
        msia_ds,
        india_ds,
        fold,
        selected_features=(),
        train_source="india",
        subsample_n=None,
    )
    return fold_data["n_train_raw"]


def run_experiment():
    """Run the size-matched arms and write the India summary tables."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    delong_rows = []

    for fold in range(N_FOLDS_CV):
        msia_ds, india_ds = load_both_cohorts()
        matched_n = india_only_train_size(msia_ds, india_ds, fold)
        print(f"\n=== Fold {fold} — matched training size N={matched_n} ===")

        india_probs = {}
        india_truth = None

        for arm_name, cfg in ARMS:
            for rep in range(N_REPEATS):
                seed = SEED + rep
                # Seed with the REPEAT's seed, not the global one:
                set_seed(seed)
                msia_ds, india_ds = load_both_cohorts()
                fold_data = prepare_fold(
                    msia_ds,
                    india_ds,
                    fold,
                    selected_features=cfg["selected"],
                    train_source=cfg["train_source"],
                    subsample_n=matched_n,
                    subsample_seed=seed,
                )
                if fold_data["country_arr"] is None:
                    print(
                        f"    [warn] fold {fold} {arm_name} rep {rep}: test rows dropped; skip"
                    )
                    continue
                model = train_lr(fold_data["train_X"], fold_data["train_Y"], seed=seed)
                prob = model.predict_proba(fold_data["test_X"])[:, 1]
                on_india = fold_data["country_arr"] == INDIA
                metrics = basic_metrics(fold_data["test_Y"][on_india], prob[on_india])
                rows.append(
                    {
                        "arm": arm_name,
                        "fold": fold,
                        "repeat": rep,
                        "matched_n": matched_n,
                        **metrics,
                    }
                )
                if rep == 0 and arm_name in ("pooled_common", "pooled_harmonized"):
                    india_probs[arm_name] = prob[on_india]
                    india_truth = fold_data["test_Y"][on_india]

        if (
            india_truth is not None
            and len(india_probs) == 2
            and len(india_probs["pooled_common"]) == len(india_probs["pooled_harmonized"])
        ):
            auc_harm, auc_common, z, p = delong_test(
                india_truth,
                india_probs["pooled_harmonized"],
                india_probs["pooled_common"],
            )
            delong_rows.append(
                {
                    "fold": fold,
                    "auc_harmonized": auc_harm,
                    "auc_common": auc_common,
                    "auc_diff": auc_harm - auc_common,
                    "z": z,
                    "p_value": p,
                }
            )

    per_repeat = pd.DataFrame(rows)
    per_repeat.to_csv(SAVE_DIR / "india_per_fold_repeat.csv", index=False)

    summary = []
    for arm_name, _ in ARMS:
        arm_rows = per_repeat[per_repeat["arm"] == arm_name]
        for metric in ("auroc", "auprc"):
            mean, lo, hi = bootstrap_ci(arm_rows[metric].tolist())
            summary.append(
                {
                    "arm": arm_name,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_obs": int(arm_rows[metric].notna().sum()),
                }
            )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(SAVE_DIR / "india_summary_ci.csv", index=False)

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(SAVE_DIR / "delong_harmonized_vs_common_india.csv", index=False)

    print("\n" + "=" * 78)
    print("SIZE-MATCHED INDIA RESULTS (mean [95% CI] across folds x repeats)")
    print("=" * 78)
    for _, row in summary_df.iterrows():
        print(
            f"  {row['arm']:<18} {row['metric']:<6} {row['mean']:.4f} "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]  (n={row['n_obs']})"
        )
    if len(delong_df):
        print("\n  DeLong (pooled_harmonized vs pooled_common) on India, per fold:")
        for _, row in delong_df.iterrows():
            print(
                f"    fold {int(row['fold'])}: dAUC={row['auc_diff']:+.4f}  "
                f"p={row['p_value']:.4f}"
            )
        print(f"    median p = {delong_df['p_value'].median():.4f}")
    print(f"\nSaved to: {SAVE_DIR}")
    return summary_df, delong_df


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
