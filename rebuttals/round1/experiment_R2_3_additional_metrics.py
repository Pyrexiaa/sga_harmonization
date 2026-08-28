"""Extended metrics, calibration and decision-curve analysis.

    Run experiment_R0_baseline_retrain.py first, then
    experiment_R0_baseline_retrain_manual.py (this script loads the
    ``sel_ute_ari_af`` weights it writes).

Run:
    python -m rebuttals.round1.experiment_R2_3_additional_metrics
"""

from __future__ import annotations

import os
import traceback
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from sga.config import (
    ECE_BINS,
    EXTERNAL_TEST_FOLD,
    HARMONIZED_SELECTED_FEATURES,
    N_FOLDS_CV,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.evaluation.calibration import apply_platt, fit_platt, reliability_curve
from sga.evaluation.dca import decision_curve_analysis, plot_dca, save_dca_csv
from sga.evaluation.metrics import METRIC_COLUMNS, expected_calibration_error, full_metrics
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA
from sga.pipeline.model_io import load_trained_model, predict_labels_and_proba

from rebuttals.round1.experiment_R0_baseline_retrain import (
    DNN_CONFIGS,
    HARMONIZED_WEIGHTS_DIR,
    build_development_folds,
    build_external_folds,
)

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND1_DIR / "R2_3_additional_metrics_and_dca"

# config name -> (family, sub-directory under the harmonized weights root, DNN config index)
MODEL_CONFIGS = [
    ("catboost_smote", "catboost", f"generalized_catboost_{SEED}", None),
    ("rf_smote", "ml", f"generalized_rf_{SEED}", None),
    ("lr_smote", "ml", f"generalized_lr_{SEED}", None),
    ("svc_smote", "ml", f"generalized_svc_{SEED}", None),
    ("stacking_smote", "ml", f"generalized_stacking_{SEED}", None),
    ("dnn_smote", "dnn", f"generalized_dnn_{SEED}_0", 0),
]

# Clean names for figure legends and titles.
DISPLAY_NAMES = {
    "catboost_smote": "CatBoost",
    "rf_smote": "RF",
    "lr_smote": "LR",
    "svc_smote": "SVC",
    "stacking_smote": "Stacking",
    "dnn_smote": "NN",
}

# Reliability bins below this count are dropped from the CURVE (they cause a spurious
# plunge on the sparse high-confidence tail once Platt has squashed the probabilities).
MIN_BIN_COUNT = 5


def _bin_counts(y_prob, n_bins=ECE_BINS):
    """Row count per equal-width probability bin, including the empty ones."""
    counts, _ = np.histogram(np.asarray(y_prob, dtype=float), bins=np.linspace(0, 1, n_bins + 1))
    return counts.astype(int)


def _curve(y_true, y_prob, n_bins=ECE_BINS):
    """``(mean predicted, observed frequency)`` for the bins we keep."""
    _, observed, predicted, _ = reliability_curve(
        y_true, y_prob, n_bins=n_bins, min_count=MIN_BIN_COUNT
    )
    return predicted, observed


def plot_reliability(model_name, y_true, p_raw, p_cal, save_dir, n_bins=ECE_BINS):
    """Reliability diagram before vs after calibration, with a bin histogram."""
    before_x, before_y = _curve(y_true, p_raw, n_bins)
    after_x, after_y = _curve(y_true, p_cal, n_bins)
    counts_before, counts_after = _bin_counts(p_raw, n_bins), _bin_counts(p_cal, n_bins)
    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0

    brier_before = brier_score_loss(y_true, p_raw)
    ece_before = expected_calibration_error(y_true, p_raw)
    brier_after = brier_score_loss(y_true, p_cal)
    ece_after = expected_calibration_error(y_true, p_cal)

    fig = plt.figure(figsize=(7, 8))
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax = fig.add_subplot(grid[0])
    ax_hist = fig.add_subplot(grid[1], sharex=ax)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax.plot(before_x, before_y, "o-", color="#d62728",
            label=f"Pre-Calibration (Brier={brier_before:.3f}, ECE={ece_before:.3f})")
    ax.plot(after_x, after_y, "s-", color="#2ca02c",
            label=f"Post-Calibration (Brier={brier_after:.3f}, ECE={ece_after:.3f})")
    ax.set_ylabel("Observed event frequency", fontsize=13)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"Reliability Diagram: {DISPLAY_NAMES.get(model_name, model_name)}\n", fontsize=13)
    plt.setp(ax.get_xticklabels(), visible=False)

    width = (1.0 / n_bins) * 0.4
    ax_hist.bar(centres - width / 2, counts_before, width=width, color="#d62728",
                alpha=0.6, label="Pre-Calibration")
    ax_hist.bar(centres + width / 2, counts_after, width=width, color="#2ca02c",
                alpha=0.6, label="Post-Calibration")
    ax_hist.set_xlabel("Mean predicted probability", fontsize=13)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.grid(alpha=0.3)
    ax_hist.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(save_dir, f"reliability_{model_name}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"mean_pred": before_x, "obs_freq": before_y}).to_csv(
        os.path.join(save_dir, f"reliability_{model_name}_before.csv"), index=False)
    pd.DataFrame({"mean_pred": after_x, "obs_freq": after_y}).to_csv(
        os.path.join(save_dir, f"reliability_{model_name}_after.csv"), index=False)
    pd.DataFrame({"bin_center": centres, "count_before": counts_before,
                  "count_after": counts_after}).to_csv(
        os.path.join(save_dir, f"reliability_{model_name}_bincounts.csv"), index=False)


def plot_dca_before_after(model_name, y_true, p_raw, p_cal, save_dir):
    """One model's decision curve before vs after calibration."""
    thresholds, net_benefit_raw, treat_all = decision_curve_analysis(y_true, p_raw)
    _, net_benefit_cal, _ = decision_curve_analysis(y_true, p_cal)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, net_benefit_raw, label="Pre-Calibration", linewidth=2, color="#d62728")
    ax.plot(thresholds, net_benefit_cal, label="Post-Calibration", linewidth=2, color="#2ca02c")
    ax.plot(thresholds, treat_all, "--", color="gray", linewidth=1, label="Treat all")
    ax.axhline(0.0, color="black", linewidth=1, label="Treat none")
    ax.set_xlabel("Threshold probability", fontsize=13)
    ax.set_ylabel("Net benefit", fontsize=13)
    ax.set_title(f"Decision Curve Comparison: {DISPLAY_NAMES.get(model_name, model_name)}",
                 fontsize=14)
    ax.set_xlim([0, 1])
    pooled = list(net_benefit_raw) + list(net_benefit_cal)
    ax.set_ylim([max(min(pooled) - 0.05, -0.5), max(pooled) + 0.05])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"dca_before_after_{model_name}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_reliability(combined, save_dir, n_bins=ECE_BINS):
    """All models on one reliability diagram: before dashed, after solid."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    cmap = plt.get_cmap("tab10")
    for index, (name, (y_true, p_raw, p_cal)) in enumerate(combined.items()):
        colour = cmap(index % 10)
        before_x, before_y = _curve(y_true, p_raw, n_bins)
        after_x, after_y = _curve(y_true, p_cal, n_bins)
        display = DISPLAY_NAMES.get(name, name)
        ax.plot(before_x, before_y, "o--", color=colour, alpha=0.55, markersize=4,
                label=f"{display} (pre)")
        ax.plot(after_x, after_y, "s-", color=colour, linewidth=2, markersize=4,
                label=f"{display} (post)")
    ax.set_xlabel("Mean predicted probability", fontsize=13)
    ax.set_ylabel("Observed event frequency", fontsize=13)
    ax.set_title("Reliability Diagram Comparison", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(save_dir, "reliability_ALL_models_before_after.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [combined] reliability diagram: {out}")


def _dca_panel(curves, thresholds, treat_all, title, out_path, xlim=(0.0, 1.0),
               restrict_y_to_xlim=False):
    """Render one multi-model decision-curve panel."""
    fig, ax = plt.subplots(figsize=(11, 7))

    def in_range(sequence):
        return [v for t, v in zip(thresholds, sequence) if xlim[0] <= t <= xlim[1]]

    pooled = []
    for label, net_benefit, colour in curves:
        ax.plot(thresholds, net_benefit, "-", color=colour, linewidth=2, label=label)
        pooled += in_range(net_benefit) if restrict_y_to_xlim else list(net_benefit)
    if treat_all is not None:
        ax.plot(thresholds, treat_all, ":", color="gray", linewidth=1, label="Treat all")
        if restrict_y_to_xlim:
            pooled += in_range(treat_all)
    ax.axhline(0.0, color="black", linewidth=1, label="Treat none")
    ax.set_xlabel("Threshold probability", fontsize=13)
    ax.set_ylabel("Net benefit", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(list(xlim))
    if pooled:
        ax.set_ylim([max(min(pooled) - 0.02, -0.5), max(pooled) + 0.02])
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [combined] decision curve: {out_path}")


def plot_combined_dca(combined, save_dir):
    """Decision curves for all models: before, after, and a zoomed-in after panel."""
    cmap = plt.get_cmap("tab10")
    thresholds = treat_all = None
    before, after = [], []
    for index, (name, (y_true, p_raw, p_cal)) in enumerate(combined.items()):
        colour = cmap(index % 10)
        display = DISPLAY_NAMES.get(name, name)
        thresholds, net_benefit_raw, treat_all = decision_curve_analysis(y_true, p_raw)
        _, net_benefit_cal, _ = decision_curve_analysis(y_true, p_cal)
        before.append((display, net_benefit_raw, colour))
        after.append((display, net_benefit_cal, colour))
    if thresholds is None:
        return
    _dca_panel(before, thresholds, treat_all, "Model Decision Curve (Pre-Calibration)",
               os.path.join(save_dir, "dca_ALL_models_before.png"))
    _dca_panel(after, thresholds, treat_all, "Model Decision Curve (Post-Calibration)",
               os.path.join(save_dir, "dca_ALL_models_after.png"))
    _dca_panel(after, thresholds, treat_all, "Model Decision Curve (Zoomed-in)",
               os.path.join(save_dir, "dca_ALL_models_after_zoom_0_0.5.png"),
               xlim=(0.0, 0.25), restrict_y_to_xlim=True)


def _country_splits(y_true, country_arr, label=""):
    """Boolean masks for the total / Malaysia / India evaluation splits."""
    splits = {"total": np.ones(len(y_true), dtype=bool)}
    if country_arr is not None and len(country_arr) == len(y_true):
        splits["malaysia"] = np.asarray(country_arr) == MALAYSIA
        splits["india"] = np.asarray(country_arr) == INDIA
    else:
        print(f"  [warn] {label}: country length mismatch; total only")
    return splits


def _summarize(frame, group_cols, metric_cols=METRIC_COLUMNS):
    """Mean +/- std across folds for each group."""
    rows = []
    for keys, subset in frame.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_folds"] = len(subset)
        for column in metric_cols:
            if column not in subset.columns:
                continue
            values = subset[column].dropna()
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_std"] = values.std()
            row[f"{column}_str"] = f"{values.mean():.4f} +/- {values.std():.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def collect_out_of_fold_predictions(config_name, family, subdir, dnn_config, development):
    """Score each fold-model on its own held-out development fold."""
    truths, probabilities = [], []
    for fold in range(N_FOLDS_CV):
        if fold not in development:
            continue
        test_X, y_true, _, features = development[fold]
        try:
            model = load_trained_model(
                family,
                str(HARMONIZED_WEIGHTS_DIR / subdir / "malaysia_tri3"),
                fold,
                n_features=len(features),
                dnn_config=dnn_config,
            )
        except Exception as error:  # noqa: BLE001
            print(f"    [calibration] fold {fold} weights unavailable: {error}")
            continue
        if model is None:
            continue
        _, y_prob = predict_labels_and_proba(model, family, test_X)
        truths.append(np.asarray(y_true))
        probabilities.append(np.asarray(y_prob))

    if not truths:
        return None, None
    return np.concatenate(truths), np.concatenate(probabilities)


def fit_development_calibrator(y_oof, p_oof):
    """Fit the Platt calibrator on development-block predictions only."""
    if y_oof is None or len(np.unique(y_oof)) < 2:
        return None
    return fit_platt(y_oof, p_oof)


def run_calibration_analysis(all_predictions, save_dir, calibrators):
    """Calibrate each model's pooled fold-4 predictions and report before vs after.

    The calibrator is fitted on the folds 0-3 out-of-fold predictions and applied
    UNCHANGED to fold 4, so the fold-4 labels never inform the calibration map.
    """
    calibration_dir = os.path.join(save_dir, "calibration")
    os.makedirs(calibration_dir, exist_ok=True)
    print("\nPlatt Scaling Calibration (Pre vs Post)")

    rows, parameters, combined = [], [], {}
    for config_name, (y_true, p_raw, country_arr) in all_predictions.items():
        y_true = np.asarray(y_true, dtype=int)
        p_raw = np.asarray(p_raw, dtype=float)
        calibrator = calibrators.get(config_name)
        if calibrator is None:
            print(f"  [{config_name}] no development calibrator; reporting raw probabilities.")
            p_cal = p_raw.copy()
        else:
            p_cal = apply_platt(calibrator, p_raw)
        combined[config_name] = (y_true, p_raw, p_cal)

        try:
            if calibrator is None:
                raise ValueError("no calibrator")
            parameters.append({
                "model": config_name,
                "platt_a": float(calibrator.coef_[0][0]),
                "platt_b": float(calibrator.intercept_[0]),
            })
        except Exception:  # noqa: BLE001 - a degenerate fold must not stop the sweep
            parameters.append({"model": config_name, "platt_a": float("nan"),
                               "platt_b": float("nan")})

        for split, mask in _country_splits(y_true, country_arr, config_name).items():
            if mask.sum() == 0:
                continue
            y_split = y_true[mask]
            for calibration, probabilities in (("before", p_raw), ("after", p_cal)):
                p_split = probabilities[mask]
                row = {"model": config_name, "eval_split": split,
                       "calibration": calibration, "n": int(mask.sum())}
                row.update(full_metrics(y_split, (p_split >= 0.5).astype(int), p_split))
                rows.append(row)

        plot_reliability(config_name, y_true, p_raw, p_cal, calibration_dir)
        plot_dca_before_after(config_name, y_true, p_raw, p_cal, calibration_dir)
        print(
            f"  {config_name:<18} "
            f"Brier {brier_score_loss(y_true, p_raw):.4f} -> {brier_score_loss(y_true, p_cal):.4f} | "
            f"ECE {expected_calibration_error(y_true, p_raw):.4f} -> "
            f"{expected_calibration_error(y_true, p_cal):.4f}"
        )

    if combined:
        plot_combined_reliability(combined, calibration_dir)
        plot_combined_dca(combined, calibration_dir)

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(calibration_dir, "calibration_metrics_by_country.csv")
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(parameters).to_csv(
        os.path.join(calibration_dir, "platt_calibrators.csv"), index=False)
    print(f"\n  Calibration metrics (total/Malaysia/India, before/after): {metrics_path}")
    print(f"  Reliability diagrams + DCA (before/after) saved under: {calibration_dir}")
    return metrics_df


def run_per_fold_calibration_external_test(per_fold_predictions, save_dir, calibrators):
    """Calibrate EACH fold-model on its own fold-4 predictions and report per fold."""
    out_dir = os.path.join(save_dir, "calibration_per_fold")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nPER-FOLD calibration evaluated on the EXTERNAL (fold-{EXTERNAL_TEST_FOLD}) test set")

    rows, parameters = [], []
    for config_name, fold_predictions in per_fold_predictions.items():
        for fold, y_true, p_raw, country_arr in sorted(fold_predictions, key=lambda t: t[0]):
            y_true = np.asarray(y_true, dtype=int)
            p_raw = np.asarray(p_raw, dtype=float)
            calibrator = calibrators.get(config_name)
            p_cal = p_raw.copy() if calibrator is None else apply_platt(calibrator, p_raw)

            try:
                if calibrator is None:
                    raise ValueError("no calibrator")
                parameters.append({
                    "model": config_name, "fold": int(fold),
                    "platt_a": float(calibrator.coef_[0][0]),
                    "platt_b": float(calibrator.intercept_[0]),
                })
            except Exception:  # noqa: BLE001
                parameters.append({"model": config_name, "fold": int(fold),
                                   "platt_a": float("nan"), "platt_b": float("nan")})

            splits = _country_splits(y_true, country_arr, f"{config_name} fold {fold}")
            for split, mask in splits.items():
                if mask.sum() == 0:
                    continue
                y_split = y_true[mask]
                for calibration, probabilities in (("before", p_raw), ("after", p_cal)):
                    p_split = probabilities[mask]
                    row = {"model": config_name, "fold": int(fold), "eval_split": split,
                           "calibration": calibration, "n": int(mask.sum())}
                    row.update(full_metrics(y_split, (p_split >= 0.5).astype(int), p_split))
                    rows.append(row)

            print(
                f"  {config_name:<16} fold {fold}:  "
                f"Brier {brier_score_loss(y_true, p_raw):.4f} -> "
                f"{brier_score_loss(y_true, p_cal):.4f} | "
                f"ECE {expected_calibration_error(y_true, p_raw):.4f} -> "
                f"{expected_calibration_error(y_true, p_cal):.4f}"
            )

    if not rows:
        print("  [per-fold calibration] no rows produced.")
        return None

    per_fold_df = pd.DataFrame(rows)
    per_fold_path = os.path.join(out_dir, "per_fold_calibrated_external_test.csv")
    per_fold_df.to_csv(per_fold_path, index=False)
    print(f"\n  Per-fold calibrated external-test metrics: {per_fold_path}")

    pd.DataFrame(parameters).to_csv(
        os.path.join(out_dir, "per_fold_platt_calibrators.csv"), index=False)

    summary_df = _summarize(per_fold_df, ("model", "eval_split", "calibration"))
    summary_path = os.path.join(out_dir, "summary_calibrated_external_test.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary (mean +/- std across folds): {summary_path}")

    print("\nPER-FOLD CALIBRATION ON EXTERNAL TEST -- total cohort, mean +/- std across folds")
    total = summary_df[summary_df["eval_split"] == "total"]
    print(f"\n{'Config':<16} {'Cal':<8} {'ROC AUC':<20} {'AUPRC':<20} {'Brier':<20} {'ECE':<20}")
    print("-" * 104)
    for _, row in total.sort_values(["model", "calibration"]).iterrows():
        print(
            f"{row['model']:<16} {row['calibration']:<8} "
            f"{row.get('roc_auc_str', 'N/A'):<20} "
            f"{row.get('auprc_str', 'N/A'):<20} "
            f"{row.get('brier_score_str', 'N/A'):<20} "
            f"{row.get('ece_str', 'N/A'):<20}"
        )
    return per_fold_df, summary_df


def run_experiment():
    """Score the pretrained harmonized models on fold 4 and report the extras."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  [manual imputation] keeping ONLY {sorted(HARMONIZED_SELECTED_FEATURES)}")
    harmonized = build_external_folds(selected_features=HARMONIZED_SELECTED_FEATURES)

    # Calibration source: each fold-model scored on its OWN held-out development
    # fold (0-3). Fitting the Platt map here, and applying it unchanged to fold 4,
    # is what keeps the fold-4 labels out of the calibration.
    development = build_development_folds(selected_features=HARMONIZED_SELECTED_FEATURES)
    calibrators = {}
    for config_name, family, subdir, dnn_index in MODEL_CONFIGS:
        dnn_config = DNN_CONFIGS[dnn_index] if dnn_index is not None else None
        y_oof, p_oof = collect_out_of_fold_predictions(
            config_name, family, subdir, dnn_config, development
        )
        calibrator = fit_development_calibrator(y_oof, p_oof)
        calibrators[config_name] = calibrator
        if calibrator is not None:
            print(f"  [calibration] {config_name}: fitted on {len(y_oof)} development rows")

    all_results = []
    all_predictions = {}      # config -> pooled (y_true, y_prob, country) on fold 4
    per_fold_predictions = {}  # config -> [(fold, y_true, y_prob, country), ...]

    for config_name, family, subdir, dnn_index in MODEL_CONFIGS:
        print(f"\nEvaluating on EXTERNAL fold-{EXTERNAL_TEST_FOLD} test: {config_name}")
        dnn_config = DNN_CONFIGS[dnn_index] if dnn_index is not None else None
        per_fold_metrics, truths, probabilities, countries = [], [], [], []

        for fold in range(N_FOLDS_CV):
            print(f"\n  --- {config_name} | fold-{fold} model on fold-{EXTERNAL_TEST_FOLD} ---")
            try:
                _, test_X, y_true, country_arr, features = harmonized[fold]
                model = load_trained_model(
                    family,
                    str(HARMONIZED_WEIGHTS_DIR / subdir / "malaysia_tri3"),
                    fold,
                    n_features=len(features),
                    dnn_config=dnn_config,
                )
                if model is None:
                    print(f"    [fold {fold}] weights missing -- skipped")
                    continue
                y_pred, y_prob = predict_labels_and_proba(model, family, test_X)

                fold_metrics = full_metrics(y_true, y_pred, y_prob)
                fold_metrics["fold"] = fold
                fold_metrics["config"] = config_name
                per_fold_metrics.append(fold_metrics)
                truths.append(np.asarray(y_true))
                probabilities.append(np.asarray(y_prob))
                countries.append(np.asarray(country_arr))
                per_fold_predictions.setdefault(config_name, []).append(
                    (fold, np.asarray(y_true), np.asarray(y_prob), np.asarray(country_arr))
                )
                print(
                    f"    BAcc={fold_metrics['balanced_accuracy']:.4f} "
                    f"AUC={fold_metrics['roc_auc']:.4f} "
                    f"F1={fold_metrics['f1']:.4f} "
                    f"AUPRC={fold_metrics['auprc']:.4f} "
                    f"Brier={fold_metrics['brier_score']:.4f} "
                    f"ECE={fold_metrics['ece']:.4f}"
                )
            except Exception as error:  # noqa: BLE001
                print(f"    [ERROR] fold {fold} failed: {error}")
                traceback.print_exc()

        if per_fold_metrics:
            all_results.extend(per_fold_metrics)
            all_predictions[config_name] = (
                np.concatenate(truths),
                np.concatenate(probabilities),
                np.concatenate(countries),
            )

    if all_predictions:
        print("\nComputing Decision Curve Analysis for all models")
        dca_results = {
            config_name: decision_curve_analysis(y_true, y_prob)
            for config_name, (y_true, y_prob, _) in all_predictions.items()
        }
        # Keys drive the legend labels; the CSV keeps the raw config names.
        plot_dca({DISPLAY_NAMES.get(k, k): v for k, v in dca_results.items()},
                 "All Models", str(SAVE_DIR), show_defaults=False)
        save_dca_csv(dca_results, "All Models", str(SAVE_DIR))

        run_calibration_analysis(all_predictions, str(SAVE_DIR), calibrators)

    if per_fold_predictions:
        run_per_fold_calibration_external_test(
            per_fold_predictions, str(SAVE_DIR), calibrators)

    if not all_results:
        print("\n[R2-3] WARNING: No metrics were collected.")
        return None

    results_df = pd.DataFrame(all_results)
    per_fold_path = SAVE_DIR / "per_fold_results.csv"
    results_df.to_csv(per_fold_path, index=False)
    print(f"\nPer-fold results saved to: {per_fold_path}")

    summary_df = _summarize(results_df, ("config",))
    summary_path = SAVE_DIR / "summary_all_experiments.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    print("\nEXTENDED METRICS SUMMARY (pretrained harmonized models)")
    print(f"\n{'Config':<24} {'Bal.Acc':<22} {'ROC AUC':<22} {'F1':<22} "
          f"{'AUPRC':<22} {'Brier':<22} {'ECE':<22}")
    print("-" * 156)
    for _, row in summary_df.iterrows():
        print(
            f"{row['config']:<24} "
            f"{row.get('balanced_accuracy_str', 'N/A'):<22} "
            f"{row.get('roc_auc_str', 'N/A'):<22} "
            f"{row.get('f1_str', 'N/A'):<22} "
            f"{row.get('auprc_str', 'N/A'):<22} "
            f"{row.get('brier_score_str', 'N/A'):<22} "
            f"{row.get('ece_str', 'N/A'):<22}"
        )
    print(f"\n{'Config':<24} {'Sensitivity':<22} {'Specificity':<22} {'PPV':<22} {'NPV':<22}")
    print("-" * 112)
    for _, row in summary_df.iterrows():
        print(
            f"{row['config']:<24} "
            f"{row.get('sensitivity_str', 'N/A'):<22} "
            f"{row.get('specificity_str', 'N/A'):<22} "
            f"{row.get('ppv_str', 'N/A'):<22} "
            f"{row.get('npv_str', 'N/A'):<22}"
        )
    print(f"\nR2-3 + R2-4 complete. Results saved to: {SAVE_DIR}")
    return results_df, summary_df


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
