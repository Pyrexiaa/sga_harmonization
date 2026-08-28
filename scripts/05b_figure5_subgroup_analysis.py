"""Manuscript Figure 5 - logistic-regression AUROC by subgroup, per cohort.

Usage:
    python -m scripts.05b_figure5_subgroup_analysis
    python -m scripts.05b_figure5_subgroup_analysis --predictions-file fold_3.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sga import config
from sga.config import set_seed
from sga.reporting.figures import (
    filter_by_country,
    plot_auroc_by_gestational_week,
    plot_auroc_by_maternal_age,
)

#: Prediction subdirectory -> legend label, in plotting order.
DEFAULT_ARMS = {
    "unified": "Unified Dataset",
    "malaysia_baseline": "Malaysia Baseline",
    "india_baseline": "India Baseline",
}

DEFAULT_COLORS = ["tab:blue", "tab:red", "tab:green"]

#: Required columns in every prediction CSV.
REQUIRED_COLUMNS = {"ga", "m_age", "Actual", "predicted_probability", "country"}


def load_arm(predictions_root, arm, predictions_file):
    """Load one arm's prediction CSV."""
    path = Path(predictions_root) / arm / predictions_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run scripts/05b0_subgroup_inference.py for the "
            f"{arm!r} arm first."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing the columns {sorted(missing)}")
    return df


def main():
    """Build the pooled and per-cohort panels of Figure 5."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--predictions-root", type=Path, default=None,
        help="root written by 05b0_subgroup_inference "
             "(default: RESULTS_DIR/figures/figure5/predictions)")
    add("--predictions-file", default="ensemble.csv",
        help="which per-arm CSV to read (ensemble.csv or fold_<n>.csv)")
    add("--arms", nargs="*", default=list(DEFAULT_ARMS),
        help="prediction subdirectories to compare, in plotting order")
    add("--labels", nargs="*", default=None, help="legend label per arm")
    add("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    set_seed(config.SEED)

    predictions_root = Path(
        args.predictions_root
        or config.results_path("figures", "figure5", "predictions")
    )
    out_dir = Path(args.output_dir or config.results_path("figures", "figure5"))
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels or [DEFAULT_ARMS.get(arm, arm) for arm in args.arms]
    if len(labels) != len(args.arms):
        raise SystemExit("--labels must give exactly one label per arm")
    colors = (DEFAULT_COLORS * len(args.arms))[: len(args.arms)]

    dfs = [load_arm(predictions_root, arm, args.predictions_file) for arm in args.arms]

    plotted = []

    def record(frame, panel):
        """Tag one panel's plotted values with the cohort it describes."""
        frame = frame.copy()
        frame.insert(0, "panel", panel)
        plotted.append(frame)

    print("\nPooled test set")
    record(plot_auroc_by_gestational_week(
        dfs, labels, out_dir / "figure5_auroc_gestational_week_pooled.png",
        colors=colors, title="AUROC per Gestational Week - Pooled",
    ), "pooled")
    record(plot_auroc_by_maternal_age(
        dfs, labels, out_dir / "figure5_auroc_maternal_age_pooled.png",
        colors=colors, title="AUROC per Maternal Age Interval - Pooled",
    ), "pooled")

    for cohort in ("malaysia", "india"):
        print(f"\n{cohort.capitalize()} subgroup")
        subsets = filter_by_country(dfs, cohort)
        name = cohort.capitalize()
        record(plot_auroc_by_gestational_week(
            subsets, labels,
            out_dir / f"figure5_auroc_gestational_week_{cohort}.png",
            colors=colors, title=f"AUROC per Gestational Week - {name}",
        ), cohort)
        record(plot_auroc_by_maternal_age(
            subsets, labels,
            out_dir / f"figure5_auroc_maternal_age_{cohort}.png",
            colors=colors, title=f"AUROC per Maternal Age Interval - {name}",
        ), cohort)

    # The panels are PNG-only otherwise, which leaves the per-stratum AUROCs and the
    # grey sample counts the Results discuss unreadable without re-running the script.
    values_csv = out_dir / "figure5_values.csv"
    pd.concat(plotted, ignore_index=True).to_csv(values_csv, index=False)

    print(f"\nFigure 5 panels and {values_csv} written to {out_dir}")


if __name__ == "__main__":
    main()
