"""Manuscript Figure 7 - AUROC against training-set size.

Reads the size-sweep summary written by
``R1.experiment_R1_2_data_scaling_inference``
(``per_country_eval_all_summary_by_country.csv``: one row per model x size x
eval_split, with ``roc_auc_mean`` and ``roc_auc_std`` across the four folds).

Writes, under ``Results/figures/figure7/``:

    figure7_combined.png/.pdf        both panels side by side, one shared legend
    figure7_auroc.png/.pdf           panel (a) alone, own legend
    figure7_pctdiff.png/.pdf         panel (b) alone, own legend
    figure7_by_cohort.png/.pdf       supplementary Total / Malaysia / India
    figure7_plotted_values.csv       the exact numbers behind the panels

Usage:
    python -m scripts.05c_figure7_sample_size              # clean lines (default)
    python -m scripts.05c_figure7_sample_size --show-sd    # overlay the +/-1 SD band
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sga import config
from sga.config import set_seed
from sga.reporting.figures import (
    FIGURE7_BASELINE_SIZE,
    figure7_plotted_values,
    load_size_summary,
    plot_figure7_auroc_only,
    plot_figure7_by_cohort,
    plot_figure7_combined,
    plot_figure7_percentage_only,
    warn_if_figure7_clipped,
)

#: Summary written by the R1-2 inference pass.
DEFAULT_SUMMARY = (
    config.ROUND1_DIR / "R1_2_data_scaling" / "per_country_eval_all_summary_by_country.csv"
)

#: Largest training size drawn by default. The full pool (18,688 combined rows) is
#: an odd point on an otherwise 1,000-step axis, so it is excluded unless asked for.
DEFAULT_MAX_SIZE = 18000

METRIC = "roc_auc"


def main():
    """Build every Figure 7 panel and the table of plotted values."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--csv", type=Path, default=DEFAULT_SUMMARY,
        help="size-sweep summary CSV (default: the R1-2 inference output)")
    add("--output-dir", type=Path, default=None)
    add("--eval-split", choices=("total", "malaysia", "india"), default="total",
        help="cohort drawn in the two main panels (Figure 7 uses the total cohort)")
    add("--max-size", type=int, default=DEFAULT_MAX_SIZE,
        help="largest training size to plot (18688 includes the full pool)")
    add("--show-sd", action="store_true",
        help="overlay the +/-1 SD across folds on the AUROC panels (off by default)")
    add("--baseline-size", type=int, default=FIGURE7_BASELINE_SIZE,
        help="training size panel (b) expresses its percentage difference against")
    add("--skip-by-cohort", action="store_true",
        help="do not render the supplementary three-panel by-cohort figure")
    args = parser.parse_args()
    set_seed(config.SEED)

    if not args.csv.exists():
        raise SystemExit(
            f"{args.csv} not found. Run the R1-2 data-scaling experiment and its "
            "per-country inference pass before building Figure 7:\n"
            "  python -m rebuttals.round1.experiment_R1_2_data_scaling --model all\n"
            "  python -m rebuttals.round1.experiment_R1_2_data_scaling_inference"
        )

    summary = load_size_summary(args.csv, metric=METRIC, max_size=args.max_size)
    if summary.empty:
        raise SystemExit(f"No rows at or below size {args.max_size} in {args.csv}.")

    out_dir = Path(args.output_dir or config.results_path("figures", "figure7"))
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.max_size == DEFAULT_MAX_SIZE else f"_to{args.max_size}"

    warn_if_figure7_clipped(
        summary, split=args.eval_split, metric=METRIC, baseline_size=args.baseline_size
    )

    plot_figure7_combined(
        summary, out_dir / f"figure7_combined{suffix}", split=args.eval_split,
        metric=METRIC, show_sd=args.show_sd, baseline_size=args.baseline_size,
    )
    plot_figure7_auroc_only(
        summary, out_dir / f"figure7_auroc{suffix}", split=args.eval_split,
        metric=METRIC, show_sd=args.show_sd,
    )
    plot_figure7_percentage_only(
        summary, out_dir / f"figure7_pctdiff{suffix}", split=args.eval_split,
        metric=METRIC, baseline_size=args.baseline_size,
    )
    if not args.skip_by_cohort:
        plot_figure7_by_cohort(
            summary, out_dir / f"figure7_by_cohort{suffix}", metric=METRIC,
            show_sd=args.show_sd,
        )

    values = figure7_plotted_values(
        summary, metric=METRIC, baseline_size=args.baseline_size
    )
    values_csv = out_dir / f"figure7_plotted_values{suffix}.csv"
    values.to_csv(values_csv, index=False)

    sizes = sorted(summary["n"].unique())
    print(f"\nx-axis: {sizes[0]:,} -> {sizes[-1]:,}  ({len(sizes)} points)")
    print(f"Figure 7 panels and {values_csv} written to {out_dir}")


if __name__ == "__main__":
    main()
