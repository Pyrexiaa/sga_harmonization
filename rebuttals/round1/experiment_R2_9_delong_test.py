"""DeLong tests between the six classifiers (Table 6).

    Run experiment_R0_baseline_retrain.py first, then
    experiment_R0_baseline_retrain_manual.py.

Run:
    python -m rebuttals.round1.experiment_R2_9_delong_test
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sga.config import (
    ALPHA,
    EXTERNAL_TEST_FOLD,
    HARMONIZED_SELECTED_FEATURES,
    N_FOLDS_CV,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.evaluation.country import append_fold_rows, evaluate_splits, save_country_results
from sga.evaluation.delong import delong_test, holm_bonferroni_correction
from sga.evaluation.metrics import EVAL_SPLITS
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA
from sga.pipeline.model_io import load_trained_model, predict_proba

from rebuttals.round1.experiment_R0_baseline_retrain import (
    DNN_CONFIGS,
    HARMONIZED_WEIGHTS_DIR,
    build_external_folds,
)

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND1_DIR / "R2_9_delong_test"

# Which trained DNN configuration enters the comparison (0, 1 or 2).
DNN_CONFIG_INDEX = 2

# Display name -> (family, weights sub-directory, DNN config index).
MODEL_LOADERS = {
    "CatBoost": ("catboost", f"generalized_catboost_{SEED}", None),
    "RandomForest": ("ml", f"generalized_rf_{SEED}", None),
    "LogisticRegression": ("ml", f"generalized_lr_{SEED}", None),
    "SVC": ("ml", f"generalized_svc_{SEED}", None),
    "Stacking": ("ml", f"generalized_stacking_{SEED}", None),
    "DNN": ("dnn", f"generalized_dnn_{SEED}_{DNN_CONFIG_INDEX}", DNN_CONFIG_INDEX),
}
MODEL_NAMES = list(MODEL_LOADERS)


def predict_proba_for(display_name, fold, test_X, n_features):
    """Load one pretrained fold-model and return P(SGA), or None if unavailable."""
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
        if model is None:
            print(f"    [skip] {display_name}: no weights for fold {fold}")
            return None
        return predict_proba(model, family, test_X)
    except Exception as error:  # noqa: BLE001 - a missing arm must not stop the sweep
        print(f"    [skip] {display_name}: load/predict failed: {error}")
        return None


def build_delong_summary(pairwise_df):
    """Aggregate the pairwise DeLong results across folds per (split, A, B)."""
    if len(pairwise_df) == 0:
        return pd.DataFrame()

    group_cols = ["eval_split", "model_A", "model_B"]
    rows = []
    for (eval_split, model_a, model_b), group in pairwise_df.groupby(group_cols):
        mean_diff = group["auc_diff"].mean()
        rows.append({
            "eval_split": eval_split,
            "model_A": model_a,
            "model_B": model_b,
            "mean_auc_A": group["auc_A"].mean(),
            "mean_auc_B": group["auc_B"].mean(),
            "mean_auc_diff": mean_diff,
            "mean_z_stat": group["z_stat"].mean(),
            "median_p_value": group["p_value"].median(),
            "min_p_value": group["p_value"].min(),
            "max_p_value": group["p_value"].max(),
            "median_p_value_holm_bonferroni": (
                group["p_value_holm_bonferroni"].median() if "p_value_holm_bonferroni" in group else float("nan")
            ),
            "max_p_value_holm_bonferroni": (
                group["p_value_holm_bonferroni"].max() if "p_value_holm_bonferroni" in group else float("nan")
            ),
            "n_folds_significant": (group["significant"] == "Yes").sum(),
            "n_folds_total": len(group),
            "conclusion": (
                f"{model_a} > {model_b}" if mean_diff > 0
                else f"{model_b} > {model_a}" if mean_diff < 0
                else "Equal"
            ),
        })
    return pd.DataFrame(rows)


def _split_masks(y_true, country_arr):
    """Boolean masks for the total / Malaysia / India evaluation splits."""
    masks = {"total": np.ones(len(y_true), dtype=bool)}
    if country_arr is not None:
        masks["malaysia"] = np.asarray(country_arr) == MALAYSIA
        masks["india"] = np.asarray(country_arr) == INDIA
    return masks


def _score_split(auc_rows, pairwise_rows, y_true, probabilities, masks, extra):
    """Append per-model AUROC rows and every pairwise DeLong row for one arm."""
    available = list(probabilities)
    for split in [s for s in EVAL_SPLITS if s in masks]:
        mask = masks[split]
        if mask.sum() == 0:
            continue
        y_split = y_true[mask]
        for name in available:
            try:
                auc = roc_auc_score(y_split, probabilities[name][mask])
            except ValueError:
                auc = float("nan")
            auc_rows.append({**extra, "eval_split": split, "model": name, "auc": auc})
        for model_a, model_b in itertools.combinations(available, 2):
            auc_a, auc_b, z_stat, p_value = delong_test(
                y_split, probabilities[model_a][mask], probabilities[model_b][mask]
            )
            pairwise_rows.append({
                **extra, "eval_split": split, "model_A": model_a, "model_B": model_b,
                "auc_A": auc_a, "auc_B": auc_b,
                "auc_diff": (auc_a - auc_b) if not np.isnan(auc_a) else np.nan,
                "z_stat": z_stat, "p_value": p_value,
                "significant": "Yes" if (not np.isnan(p_value) and p_value < ALPHA) else "No",
            })


def _apply_holm_bonferroni(pairwise_df, group_cols):
    """Add Holm-Bonferroni adjusted p-values within each comparison family.

    The family is the set of pairwise comparisons sharing ``group_cols`` - for
    manuscript Table 6 that is the 15 pairs within one ``eval_split``.
    """
    if pairwise_df.empty:
        return pairwise_df

    present = [c for c in group_cols if c in pairwise_df.columns]
    pairwise_df = pairwise_df.copy()
    pairwise_df["p_value_holm_bonferroni"] = np.nan

    grouped = pairwise_df.groupby(present, dropna=False) if present else [((), pairwise_df)]
    for _, group in grouped:
        pairwise_df.loc[group.index, "p_value_holm_bonferroni"] = holm_bonferroni_correction(
            group["p_value"].to_numpy()
        )

    pairwise_df["n_comparisons_in_family"] = pairwise_df.groupby(
        present, dropna=False
    )["p_value"].transform("size") if present else len(pairwise_df)
    pairwise_df["significant"] = np.where(
        pairwise_df["p_value_holm_bonferroni"].notna() & (pairwise_df["p_value_holm_bonferroni"] < ALPHA),
        "Yes", "No",
    )
    return pairwise_df


def run_experiment():
    """Compare every model pair on the external fold with DeLong's test."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    harmonized = build_external_folds(selected_features=HARMONIZED_SELECTED_FEATURES)
    y_true = harmonized[0][2]
    country_arr = harmonized[0][3]
    masks = _split_masks(y_true, country_arr)
    print(
        f"Fold-{EXTERNAL_TEST_FOLD} external test -- "
        f"Malaysia: {int(masks['malaysia'].sum())}, India: {int(masks['india'].sum())}"
    )

    pairwise_rows, auc_rows, metrics_rows = [], [], []
    ensemble_probabilities = {name: [] for name in MODEL_NAMES}

    for fold in range(N_FOLDS_CV):
        print(f"\nFold-{fold} model  ->  fold-{EXTERNAL_TEST_FOLD} external test")
        _, test_X, _, _, features = harmonized[fold]

        fold_probabilities = {}
        for name in MODEL_NAMES:
            probabilities = predict_proba_for(name, fold, test_X, len(features))
            if probabilities is not None:
                fold_probabilities[name] = probabilities
                ensemble_probabilities[name].append(probabilities)

        for name, probabilities in fold_probabilities.items():
            append_fold_rows(
                metrics_rows,
                evaluate_splits(
                    y_true, (probabilities >= 0.5).astype(int), probabilities, country_arr
                ),
                name, fold,
            )

        _score_split(auc_rows, pairwise_rows, y_true, fold_probabilities, masks,
                     extra={"fold": fold})

    pairwise_df = _apply_holm_bonferroni(pd.DataFrame(pairwise_rows), ("fold", "eval_split"))
    pairwise_df.to_csv(SAVE_DIR / "delong_pairwise_per_fold.csv", index=False)
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(SAVE_DIR / "auc_per_model_per_fold.csv", index=False)
    if metrics_rows:
        save_country_results(metrics_rows, str(SAVE_DIR), prefix="model_metrics",
                             group_cols=("model", "eval_split"))

    summary_df = build_delong_summary(pairwise_df)
    summary_df.to_csv(SAVE_DIR / "delong_summary.csv", index=False)

    auc_summary = (
        auc_df.groupby(["eval_split", "model"])
        .agg(mean_auc=("auc", "mean"), std_auc=("auc", "std"), n_folds=("auc", "count"))
        .reset_index()
    )
    auc_summary["auc_str"] = auc_summary.apply(
        lambda r: f"{r['mean_auc']:.4f} +/- {r['std_auc']:.4f}", axis=1)
    auc_summary.to_csv(SAVE_DIR / "auc_ranking.csv", index=False)

    # Ensemble: mean of the four fold-models' probabilities on the same rows.
    ensemble_pairwise, ensemble_auc = [], []
    ensemble = {
        name: np.mean(np.vstack(stack), axis=0)
        for name, stack in ensemble_probabilities.items() if stack
    }
    _score_split(ensemble_auc, ensemble_pairwise, y_true, ensemble, masks, extra={})
    ensemble_pairwise_df = _apply_holm_bonferroni(pd.DataFrame(ensemble_pairwise), ("eval_split",))
    pd.DataFrame(ensemble_auc).to_csv(SAVE_DIR / "auc_ensemble_fold4.csv", index=False)
    ensemble_pairwise_df.to_csv(SAVE_DIR / "delong_ensemble_fold4.csv", index=False)

    for split in EVAL_SPLITS:
        split_auc = auc_summary[auc_summary["eval_split"] == split]
        if len(split_auc) == 0:
            continue
        split_summary = (
            summary_df[summary_df["eval_split"] == split] if len(summary_df) else summary_df
        )
        split_auc = split_auc.sort_values("mean_auc", ascending=False).reset_index(drop=True)
        print(f"\nDELONG on FOLD-{EXTERNAL_TEST_FOLD} external test -- per fold-model "
              f"-- split: {split.upper()} (alpha={ALPHA})")
        print("  AUC ranking (mean +/- std across the 4 fold-models):")
        for rank, (_, row) in enumerate(split_auc.iterrows(), 1):
            print(f"    {rank}. {row['model']:<20} {row['auc_str']}")
        for _, row in split_summary.iterrows():
            print(f"  {row['model_A']:<18} vs {row['model_B']:<18} "
                  f"AUC {row['mean_auc_A']:.4f}/{row['mean_auc_B']:.4f} "
                  f"med.p={row['median_p_value']:.4f} "
                  f"sig={row['n_folds_significant']}/{row['n_folds_total']}")

    if ensemble_auc:
        print(f"\nENSEMBLE (mean of {N_FOLDS_CV} fold-models) on "
              f"FOLD-{EXTERNAL_TEST_FOLD} external test")
        ensemble_auc_df = pd.DataFrame(ensemble_auc)
        for split in EVAL_SPLITS:
            subset = ensemble_auc_df[ensemble_auc_df["eval_split"] == split]
            subset = subset.sort_values("auc", ascending=False)
            if len(subset) == 0:
                continue
            print(f"  [{split}] " + ", ".join(
                f"{r['model']}={r['auc']:.4f}" for _, r in subset.iterrows()))

    print(f"\nExperiment complete. Results saved to: {SAVE_DIR}")
    return pairwise_df, summary_df, auc_summary


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
