"""Manuscript Figure 3 - AUROC by classifier and training strategy, per cohort.

Usage:
    python -m scripts.05a_figure3_auroc_comparison
    python -m scripts.05a_figure3_auroc_comparison --cohorts malaysia
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sga import config
from sga.config import set_seed
from sga.evaluation.bootstrap import bootstrap_ci
from sga.reporting.figures import MODEL_ORDER, STRATEGY_LABELS, plot_auroc_comparison

#: Per-fold result file written by ``run_external_inference``.
DEFAULT_RESULTS_FILE = f"external_fold{config.EXTERNAL_TEST_FOLD}_per_fold_results.csv"

#: Directory-name templates per strategy, relative to ``--results-root``.
DEFAULT_TEMPLATES = {
    "baseline": "baseline_{cohort}_{model}_{seed}",
    "unified": "unified_{model}_{seed}",
    "cross_domain": "single_source_{other}/generalized_{model}_common_{seed}/malaysia_tri3",
}

#: The opposite cohort, for the ``{other}`` placeholder.
OTHER_COHORT = {"malaysia": "india", "india": "malaysia"}

#: Strategies whose run directories live under the round-1 experiment tree rather than
#: under ``--results-root``.
ALTERNATE_ROOTS = {"cross_domain": config.ROUND1_DIR / "R0_baseline_retrain"}

#: The cross-domain (single-source) DNN weights are stored one directory per DNN
#: configuration, so the ``{model}`` token has to name a configuration for that arm.
CROSS_DOMAIN_DNN_CONFIG = 0


def template_model_token(strategy, model, dnn_config=CROSS_DOMAIN_DNN_CONFIG):
    """The ``{model}`` token to interpolate into a strategy's directory template."""
    if strategy == "cross_domain" and model == "dnn":
        return f"dnn_cfg{dnn_config}"
    return model


def read_fold_auroc(results_csv, eval_split):
    """Read the per-fold AUROC values of one cohort from a results CSV."""
    results_csv = Path(results_csv)
    if not results_csv.exists():
        return []
    df = pd.read_csv(results_csv)
    if not {"eval_split", "roc_auc"} <= set(df.columns):
        return []
    return df.loc[df["eval_split"] == eval_split, "roc_auc"].dropna().tolist()


def _strategy_root(strategy, default_root):
    """Directory the runs of ``strategy`` live under."""
    return Path(ALTERNATE_ROOTS.get(strategy, default_root))


def collect_records(root, cohort, templates, results_file, seed, models,
                    dnn_config=CROSS_DOMAIN_DNN_CONFIG):
    """Assemble the mean AUROC and 95% CI of every strategy/classifier pair."""
    records = []
    for strategy, template in templates.items():
        for model in models:
            directory = _strategy_root(strategy, root) / template.format(
                cohort=cohort,
                other=OTHER_COHORT[cohort],
                model=template_model_token(strategy, model, dnn_config),
                seed=seed,
            )
            values = read_fold_auroc(directory / results_file, cohort)
            if not values:
                print(f"  [missing] {strategy}/{model}: {directory / results_file}")
                continue
            mean, low, high = bootstrap_ci(values)
            records.append({
                "cohort": cohort,
                "model": model,
                "strategy": strategy,
                "auroc": mean,
                "ci_low": low,
                "ci_high": high,
                "n_folds": len(values),
            })
    return records


def main():
    """Build Figure 3 for each requested cohort and save the plotted values."""
    parser = argparse.ArgumentParser(description=__doc__)
    add = parser.add_argument
    add("--results-root", type=Path, default=config.MODEL_DIR)
    add("--output-dir", type=Path, default=None)
    add("--cohorts", nargs="*", choices=("malaysia", "india"),
        default=["malaysia", "india"])
    add("--models", nargs="*", choices=MODEL_ORDER, default=MODEL_ORDER)
    add("--results-file", default=DEFAULT_RESULTS_FILE)
    add("--seed", type=int, default=config.SEED)
    add("--cross-domain-dnn-config", type=int, default=CROSS_DOMAIN_DNN_CONFIG,
        help="which single-source DNN configuration represents the cross-domain arm")
    add("--allow-missing-strategies", action="store_true",
        help="render the figure even when one of the three arms has no results. "
             "Off by default: a silently two-armed Figure 3 looks complete but is not.")
    for strategy in STRATEGY_LABELS:
        add(f"--{strategy.replace('_', '-')}-template",
            default=DEFAULT_TEMPLATES[strategy],
            help=f"directory template for the {strategy} arm")
    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir or config.results_path("figures", "figure3"))
    out_dir.mkdir(parents=True, exist_ok=True)
    templates = {
        strategy: getattr(args, f"{strategy}_template") for strategy in STRATEGY_LABELS
    }

    all_records = []
    for cohort in args.cohorts:
        print(f"\nCollecting {cohort} results from {args.results_root}")
        records = collect_records(
            args.results_root, cohort, templates, args.results_file,
            args.seed, args.models, args.cross_domain_dnn_config,
        )
        if not records:
            print(f"  no results found for {cohort}; skipping the panel")
            continue
        all_records.extend(records)
        plot_auroc_comparison(
            records, out_dir / f"figure3_auroc_comparison_{cohort}.png", cohort=cohort
        )

    if not all_records:
        raise SystemExit(
            "No per-fold results found. Run the 03* training and 04* inference "
            "scripts before building Figure 3."
        )

    found = {record["strategy"] for record in all_records}
    missing = [strategy for strategy in STRATEGY_LABELS if strategy not in found]
    if missing:
        hint = {
            "baseline": "python scripts/03d_train_country_baseline.py + 04d (per cohort "
                        "and model)",
            "unified": "python scripts/03a-03c + 04a-04c",
            "cross_domain": "python -c \"from rebuttals.round1."
                            "experiment_R0_baseline_retrain import *; "
                            "retrain_single_source_common_all()\" then "
                            "python -m rebuttals.round1.experiment_R0_baseline_retrain",
        }
        message = "\n".join(
            [f"Figure 3 is missing the {s!r} arm -- produce it with: {hint[s]}"
             for s in missing]
        )
        if not args.allow_missing_strategies:
            raise SystemExit(
                message
                + "\n\nFigure 3's caption describes three arms per cohort. Pass "
                  "--allow-missing-strategies to render a partial figure anyway."
            )
        print("\nWARNING: rendering a partial Figure 3.\n" + message)
    values_csv = out_dir / "figure3_auroc_values.csv"
    pd.DataFrame(all_records).to_csv(values_csv, index=False)
    print(f"\nFigure 3 panels and {values_csv} written to {out_dir}")


if __name__ == "__main__":
    main()
