"""Cohort-level disparity with uncertainty.

Produces the manuscript's "Cohort-Level Fairness Analysis" section and appendix
Table S5. Everything here is computed on the leakage-free external test fold using
the Platt-CALIBRATED probabilities, which is what the section states:

* Per-cohort rates get Wilson score intervals; between-cohort differences get
  Newcombe hybrid score intervals. These are the reported values.
* A cohort-stratified pregnancy-cluster bootstrap is reported ALONGSIDE, so the
  claim that the score intervals are better behaved at these event counts can be
  checked rather than asserted.
* Differences are signed (Malaysia minus India) across the whole 0.10-0.90 grid,
  because a threshold-dependent disparity is not a fixed property of the model.

Two operating points carry the narrative: the default 0.50 cut-off, retained for
comparability with Tables 4 and 5 but explicitly not a screening operating point,
and the 0.10 screening threshold chosen a priori from the 5-20% clinically relevant
range identified by the decision-curve analysis.

Run:
    python -m rebuttals.round2.experiment_R2_5_fairness_uncertainty
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sga.config import (
    DECISION_THRESHOLD,
    MIN_POSITIVE_PREDICTIONS_FOR_PPV,
    N_BOOTSTRAP,
    ROUND2_DIR,
    SCREENING_THRESHOLD,
    SEED,
    THRESHOLD_GRID,
    set_seed,
)
from sga.evaluation.bootstrap import bootstrap_metric_ci
from sga.evaluation.fairness import (
    bootstrap_fairness_sweep,
    fairness_threshold_sweep,
)
from sga.pipeline.external_fold import build_calibrated_external_fold
from sga.reporting.figures import plot_fairness_vs_threshold

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND2_DIR / "R2_5_fairness_uncertainty"

#: The four differences quoted in the section, in the column order of Table S5.
TABLE_S5_DIFFERENCES = [
    ("equal_opportunity_diff", "Equal Opportunity Difference"),
    ("equalized_odds_diff", "Equalized Odds Difference"),
    ("demographic_parity_diff", "Demographic Parity Difference"),
    ("predictive_parity_diff", "Predictive Parity Difference"),
]

#: Per-cohort rates tabulated under each operating point.
COHORT_RATE_LABELS = [
    ("tpr", "True-positive rate (sensitivity)"),
    ("tnr", "Specificity"),
    ("fpr", "False-positive rate"),
    ("ppv", "PPV"),
    ("npv", "NPV"),
    ("pred_rate", "Positive prediction rate"),
]


def build_table_s5(sweep, bootstrap_sweep=None):
    """Appendix Table S5: one row per threshold, one column per difference."""
    merged = sweep
    if bootstrap_sweep is not None:
        merged = sweep.merge(bootstrap_sweep, on="threshold", how="left")

    rows = []
    for _, record in merged.iterrows():
        row = {"threshold": f"{record['threshold']:.2f}"}
        for key, label in TABLE_S5_DIFFERENCES:
            flag = ""
            if key == "predictive_parity_diff":
                flag = record.get("predictive_parity_diff_flag", "")
            if flag == "N/A":
                row[f"{label} (95% CI)"] = "N/A"
            else:
                point = record[f"{key}_abs"] if key == "equalized_odds_diff" else record[key]
                if np.isnan(point):
                    row[f"{label} (95% CI)"] = "N/A"
                else:
                    row[f"{label} (95% CI)"] = (
                        f"{point:.4f}{flag} "
                        f"({record[f'{key}_ci_low']:.4f} - {record[f'{key}_ci_high']:.4f})"
                    )
            if bootstrap_sweep is not None and f"{key}_boot_ci_low" in record:
                low, high = record[f"{key}_boot_ci_low"], record[f"{key}_boot_ci_high"]
                row[f"{label} [bootstrap 95% CI]"] = (
                    "N/A" if np.isnan(low) else f"({low:.4f} - {high:.4f})"
                )
        rows.append(row)
    return pd.DataFrame(rows)


def discrimination_by_cohort(external, n_boot=N_BOOTSTRAP):
    """AUROC, AUPRC, ECE and Brier per cohort with pregnancy-cluster bootstrap CIs.

    These four are not proportions, so the bootstrap - not a score interval - is
    the method the Statistical Analysis section prescribes for them.
    """
    rows = []
    for name, mask in external.splits():
        clusters = (
            external.cluster_ids[mask] if external.cluster_ids is not None else None
        )
        row = {
            "split": name,
            "n": int(mask.sum()),
            "n_pos": int(external.y_true[mask].sum()),
            "prevalence": float(external.y_true[mask].mean()),
        }
        for metric in ("auroc", "auprc", "ece", "brier"):
            point, low, high = bootstrap_metric_ci(
                external.y_true[mask],
                external.p_calibrated[mask],
                metric=metric,
                n_boot=n_boot,
                cluster_ids=clusters,
            )
            row[metric] = point
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_str"] = f"{point:.4f} ({low:.4f} - {high:.4f})"
        rows.append(row)
    return pd.DataFrame(rows)


def _operating_point_block(record, lines):
    """Append the narrative block for one threshold to ``lines``."""
    lines.append(f"Operating point: calibrated probability >= {record['threshold']:.2f}")
    lines.append("  Confusion counts")
    for cohort in ("malaysia", "india"):
        lines.append(
            f"    {cohort.capitalize():<9} TP={int(record[f'tp_{cohort}'])} "
            f"FP={int(record[f'fp_{cohort}'])} FN={int(record[f'fn_{cohort}'])} "
            f"TN={int(record[f'tn_{cohort}'])}"
        )
    lines.append("  Per-cohort rates (Wilson 95% CI)")
    for key, label in COHORT_RATE_LABELS:
        lines.append(f"    {label}")
        for cohort in ("malaysia", "india"):
            lines.append(
                f"      {cohort.capitalize():<9} {record[f'{key}_{cohort}_str']}"
            )
    lines.append("  Differences, signed Malaysia minus India (Newcombe 95% CI)")
    for key, label in TABLE_S5_DIFFERENCES:
        excludes = record.get(f"{key}_ci_excludes_0", False)
        flag = record.get("predictive_parity_diff_flag", "") if "predictive" in key else ""
        note = " [not reliably estimable]" if flag in ("*", "N/A") else ""
        lines.append(
            f"    {label:<32} {record[f'{key}_str']}   "
            f"|{abs(record[key]):.4f}|   "
            f"{'CI excludes 0' if excludes else 'CI includes 0'}{note}"
        )
    lines.append("")


def write_manuscript_block(external, sweep, discrimination, path):
    """Ready-to-check numbers for the Cohort-Level Fairness section."""
    lines = ["=" * 78]
    lines.append("COHORT-LEVEL FAIRNESS - calibrated probabilities, external test fold")
    lines.append(
        "Rates: Wilson score 95% CI.  Differences: Newcombe hybrid score 95% CI."
    )
    lines.append("=" * 78)
    lines.append("")
    lines.append("Test-fold composition")
    lines.append(external.composition().to_string(index=False))
    lines.append("")

    for threshold in (DECISION_THRESHOLD, SCREENING_THRESHOLD):
        match = sweep[np.isclose(sweep["threshold"], threshold)]
        if match.empty:
            continue
        _operating_point_block(match.iloc[0], lines)

    lines.append("Discrimination and calibration by cohort (bootstrap 95% CI)")
    for _, record in discrimination.iterrows():
        lines.append(
            f"  {record['split']:<9} n={int(record['n']):<5} SGA={int(record['n_pos']):<4} "
            f"AUROC={record['auroc_str']}  AUPRC={record['auprc_str']}"
        )
        lines.append(f"  {'':<9} ECE={record['ece_str']}  Brier={record['brier_str']}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("Threshold dependence of the Equal Opportunity Difference")
    lines.append("(the disparity is NOT a fixed property of the model - report the range)")
    lines.append("-" * 78)
    lines.append(f"  {'thr':>5}  {'TPR_MY':>7}  {'TPR_IN':>7}  {'EOD (signed)':>13}")
    for _, record in sweep.iterrows():
        lines.append(
            f"  {record['threshold']:>5.2f}  {record['tpr_malaysia']:>7.4f}  "
            f"{record['tpr_india']:>7.4f}  {record['equal_opportunity_diff']:>+13.4f}"
        )

    text = "\n".join(lines)
    path.write_text(text + "\n")
    print(text)
    return text


def run_experiment(n_boot=N_BOOTSTRAP):
    """Compute the fairness sweep, its intervals and the manuscript block."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    external = build_calibrated_external_fold()
    if not external.has_cohorts:
        raise SystemExit(
            "The Malaysia/India split of the external fold is unavailable "
            "(the physiological filter dropped test rows), so no cohort-level "
            "fairness metric can be computed."
        )
    external.composition().to_csv(SAVE_DIR / "test_fold_composition.csv", index=False)

    print(
        f"\nSweeping {len(THRESHOLD_GRID)} thresholds with Wilson/Newcombe intervals, "
        f"and {n_boot} bootstrap replicates over "
        f"{external.bootstrap_unit()}, stratified by cohort..."
    )
    sweep = fairness_threshold_sweep(
        external.y_true,
        external.p_calibrated,
        external.country_arr,
        thresholds=THRESHOLD_GRID,
        min_positive_predictions=MIN_POSITIVE_PREDICTIONS_FOR_PPV,
    )
    bootstrap = bootstrap_fairness_sweep(
        external.y_true,
        external.p_calibrated,
        external.country_arr,
        thresholds=THRESHOLD_GRID,
        cluster_ids=external.cluster_ids,
        n_boot=n_boot,
    )

    sweep.to_csv(SAVE_DIR / "fairness_calibrated_threshold_sweep.csv", index=False)
    bootstrap.to_csv(SAVE_DIR / "fairness_bootstrap_ci.csv", index=False)

    table_s5 = build_table_s5(sweep, bootstrap)
    table_s5.to_csv(SAVE_DIR / "tableS5_cohort_fairness.csv", index=False)

    for threshold, name in (
        (DECISION_THRESHOLD, "default_0.50"),
        (SCREENING_THRESHOLD, "screening_0.10"),
    ):
        subset = sweep[np.isclose(sweep["threshold"], threshold)]
        subset.to_csv(SAVE_DIR / f"fairness_at_{name}.csv", index=False)

    discrimination = discrimination_by_cohort(external, n_boot=n_boot)
    discrimination.to_csv(SAVE_DIR / "discrimination_by_cohort_ci.csv", index=False)

    write_manuscript_block(
        external, sweep, discrimination, SAVE_DIR / "fairness_manuscript_numbers.txt"
    )
    try:
        plot_fairness_vs_threshold(sweep, SAVE_DIR / "fairness_vs_threshold.png")
    except Exception as error:  # noqa: BLE001 - a missing figure must not lose the CSVs
        print(f"  [fairness plot skipped: {error}]")

    print(f"\nSaved to: {SAVE_DIR}")
    return sweep, discrimination


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
