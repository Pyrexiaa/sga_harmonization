"""Decision-curve analysis (manuscript Figure 4)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sga.config import DCA_CLINICAL_RANGE, DCA_THRESHOLD_RANGE, MODEL_DISPLAY_NAMES


def decision_curve_analysis(y_true, y_prob, thresholds=None):
    """Net benefit of the model and of the treat-all strategy."""
    if thresholds is None:
        # config.DCA_THRESHOLD_RANGE is the range reported in the manuscript.
        low, high = DCA_THRESHOLD_RANGE
        thresholds = np.arange(max(low, 0.01), min(high, 0.99) + 1e-9, 0.01)

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    prevalence = y_true.mean()

    net_benefits, treat_all = [], []
    for threshold in thresholds:
        predicted = (y_prob >= threshold).astype(int)
        tp = np.sum((predicted == 1) & (y_true == 1))
        fp = np.sum((predicted == 1) & (y_true == 0))
        odds = threshold / (1 - threshold)
        net_benefits.append(tp / n - (fp / n) * odds)
        treat_all.append(prevalence - (1 - prevalence) * odds)

    return thresholds, net_benefits, treat_all


def _safe_name(scenario_name):
    return scenario_name.replace(" ", "_").replace(",", "").replace(":", "").lower()


def plot_dca(dca_results, scenario_name, save_dir, show_defaults=True, zoom=False):
    """Plot decision curves for several models."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    treat_all_curve = None
    for model_name, (thresholds, net_benefits, treat_all) in dca_results.items():
        ax.plot(
            thresholds,
            net_benefits,
            label=MODEL_DISPLAY_NAMES.get(model_name, model_name),
            linewidth=2,
        )
        treat_all_curve = (thresholds, treat_all)

    if show_defaults and treat_all_curve is not None:
        thresholds, treat_all = treat_all_curve
        ax.plot(thresholds, treat_all, "k--", linewidth=1.5, label="Treat all")
        ax.axhline(0, color="grey", linewidth=1.5, linestyle=":", label="Treat none")

    ax.set_xlabel("Threshold probability", fontsize=14)
    ax.set_ylabel("Net benefit", fontsize=14)
    ax.set_title(f"Decision curve analysis: {scenario_name}", fontsize=16)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    all_net_benefits = [v for _, nb, _ in dca_results.values() for v in nb]
    if zoom:
        # Figure 4(b): the clinically relevant 5-20% range.
        ax.set_xlim(DCA_CLINICAL_RANGE)
    else:
        # Figure 4(a): exactly the thresholds that were computed. Drawing out to
        # 1.0 left the right-hand two thirds of the panel empty and implied a
        # range the curves do not cover.
        ax.set_xlim(*DCA_THRESHOLD_RANGE)
    ax.set_ylim(max(min(all_net_benefits) - 0.05, -0.5), max(all_net_benefits) + 0.05)

    plt.tight_layout()
    stem = os.path.join(save_dir, f"dca_{_safe_name(scenario_name)}{'_zoom' if zoom else ''}")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"{stem}.png"


def save_dca_csv(dca_results, scenario_name, save_dir):
    """Persist the net-benefit curves so the figure can be regenerated."""
    os.makedirs(save_dir, exist_ok=True)
    rows = [
        {
            "model": model_name,
            "threshold": threshold,
            "net_benefit": net_benefits[i],
            "treat_all_benefit": treat_all[i],
            "treat_none_benefit": 0.0,
        }
        for model_name, (thresholds, net_benefits, treat_all) in dca_results.items()
        for i, threshold in enumerate(thresholds)
    ]
    path = os.path.join(save_dir, f"dca_data_{_safe_name(scenario_name)}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
