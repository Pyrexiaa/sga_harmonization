"""Per-cohort evaluation and result persistence."""

from __future__ import annotations

import os

import pandas as pd

from sga.evaluation.metrics import EVAL_SPLITS, METRIC_COLUMNS, full_metrics
from sga.config import INDIA, MALAYSIA

import numpy as np


def _empty_row():
    return {k: float("nan") for k in METRIC_COLUMNS} | {"n_samples": 0}


def evaluate_splits(y_true, y_pred, y_prob, country_arr):
    """Metrics on the total, Malaysia-only and India-only subsets."""
    y_true, y_pred, y_prob = (np.asarray(a) for a in (y_true, y_pred, y_prob))
    country_arr = np.asarray(country_arr)

    if len(y_true) != len(country_arr):
        raise ValueError(
            f"country_arr length ({len(country_arr)}) must match the number of test "
            f"samples ({len(y_true)}); check the Malaysia-first row ordering."
        )

    results = {"total": full_metrics(y_true, y_pred, y_prob)}
    for name, code in (("malaysia", MALAYSIA), ("india", INDIA)):
        mask = country_arr == code
        results[name] = (
            full_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            if mask.sum() > 0
            else _empty_row()
        )
    return results


def append_fold_rows(all_results, split_metrics, model_name, fold, extra=None):
    """Append one row per split to a running results list."""
    for split_name in EVAL_SPLITS:
        row = {"model": model_name, "fold": fold, "eval_split": split_name}
        if extra:
            row.update(extra)
        row.update(split_metrics[split_name])
        all_results.append(row)
    return all_results


def summarize(results_df, group_cols=("model", "eval_split")):
    """Mean +/- std across folds for each group."""
    group_cols = [c for c in group_cols if c in results_df.columns] or ["eval_split"]

    rows = []
    for keys, subset in results_df.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        for col in METRIC_COLUMNS:
            if col not in subset.columns:
                continue
            values = subset[col].dropna()
            if len(values):
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_std"] = values.std()
                row[f"{col}_str"] = f"{values.mean():.4f} +/- {values.std():.4f}"
            else:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
                row[f"{col}_str"] = "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def save_country_results(
    all_results, save_dir, prefix="country", model_name=None,
    group_cols=("model", "eval_split"),
):
    """Write per-fold and summary CSVs and print a compact table."""
    os.makedirs(save_dir, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print(f"[country_eval] No results to save in {save_dir}")
        return results_df, results_df

    results_df = results_df.drop_duplicates().reset_index(drop=True)
    per_fold_path = os.path.join(save_dir, f"{prefix}_per_fold_results.csv")
    results_df.to_csv(per_fold_path, index=False)

    summary_df = summarize(results_df, group_cols=group_cols)
    summary_path = os.path.join(save_dir, f"{prefix}_summary_by_country.csv")
    summary_df.to_csv(summary_path, index=False)

    header = f"PER-COUNTRY TEST METRICS{f' [{model_name}]' if model_name else ''}"
    print("\n" + "=" * 110)
    print(header)
    print("=" * 110)
    print(f"  {'split':<10} {'Bal.Acc':<22} {'ROC AUC':<22} {'F1':<22} {'Sens.':<22} {'Spec.':<22}")
    print("  " + "-" * 104)
    for _, row in summary_df.iterrows():
        print(
            f"  {str(row.get('eval_split', '')):<10} "
            f"{row.get('balanced_accuracy_str', 'N/A'):<22} "
            f"{row.get('roc_auc_str', 'N/A'):<22} "
            f"{row.get('f1_str', 'N/A'):<22} "
            f"{row.get('sensitivity_str', 'N/A'):<22} "
            f"{row.get('specificity_str', 'N/A'):<22}"
        )
    print(f"\n  Saved: {per_fold_path}\n  Saved: {summary_path}")
    return results_df, summary_df
