"""Calibration and operating points with uncertainty.

Produces manuscript Table 3 (ECE and Brier with 95% CIs for the Logistic
Regression on the total, Malaysian and Indian cohorts) and the operating points
quoted in "Model Calibration and Performances".

Which interval belongs to which quantity, per the Statistical Analysis section:

* ECE and Brier are not proportions, so they get percentile bootstrap intervals
  (2,000 iterations). Where a pregnancy identifier is available the bootstrap
  resamples whole pregnancies, matching the cluster-aware analysis.
* Sensitivity, specificity, PPV and NPV are proportions, so they get Wilson score
  intervals - stable at the Indian stratum's event counts, where a percentile
  bootstrap is not.

Operating points. The primary threshold-dependent results use the fixed 0.50
cut-off. The two alternatives reported alongside - Youden's J and the highest
threshold that still reaches sensitivity >= 0.80 - are selected on the DEVELOPMENT
block's out-of-fold predictions and only then applied to fold 4, which is what the
Methods require of any alternative operating point. The 0.10 screening threshold
used by the fairness section is evaluated here too, so all four appear on one
probability scale.

Run:
    python -m rebuttals.round2.experiment_R2_3_calibration_uncertainty
"""

from __future__ import annotations

import warnings

import pandas as pd

from sga.config import (
    DECISION_THRESHOLD,
    N_BOOTSTRAP,
    ROUND2_DIR,
    SCREENING_THRESHOLD,
    SEED,
    set_seed,
)
from sga.evaluation.bootstrap import bootstrap_metric_ci
from sga.evaluation.metrics import confusion_counts, rate_numerator_denominator
from sga.evaluation.proportions import rate_with_ci
from sga.pipeline.external_fold import TARGET_SENSITIVITY, build_calibrated_external_fold

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND2_DIR / "R2_3_calibration_uncertainty"

#: Threshold-dependent rates reported at each operating point.
OPERATING_POINT_RATES = ("sensitivity", "specificity", "ppv", "npv")


def _operating_point_columns(y_true, y_prob, threshold, suffix):
    """Confusion counts and Wilson-bounded rates at one cut-off, as flat columns."""
    counts = confusion_counts(y_true, (y_prob >= threshold).astype(int))
    columns = {f"{key}@{suffix}": value for key, value in counts.items()}
    for rate in OPERATING_POINT_RATES:
        numerator, denominator = rate_numerator_denominator(counts, rate)
        interval = rate_with_ci(numerator, denominator)
        columns[f"{rate}@{suffix}"] = interval.point
        columns[f"{rate}@{suffix}_ci"] = interval.format()
    return columns


def run_experiment(n_boot=N_BOOTSTRAP):
    """Evaluate calibration and operating points on the external test fold."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    external = build_calibrated_external_fold()
    external.composition().to_csv(SAVE_DIR / "test_fold_composition.csv", index=False)

    operating_points = [
        (DECISION_THRESHOLD, "0.5"),
        (SCREENING_THRESHOLD, "screening"),
        (external.youden_threshold, "youden"),
        (external.sensitivity_threshold, "target"),
    ]

    rows = []
    for split, mask in external.splits():
        y_split = external.y_true[mask]
        p_split = external.p_calibrated[mask]
        clusters = (
            external.cluster_ids[mask] if external.cluster_ids is not None else None
        )

        row = {
            "split": split,
            "n": int(mask.sum()),
            "n_pos": int(y_split.sum()),
            "prevalence": float(y_split.mean()),
        }
        for metric in ("ece", "brier"):
            point, low, high = bootstrap_metric_ci(
                y_split, p_split, metric=metric, n_boot=n_boot, cluster_ids=clusters
            )
            row[metric] = point
            row[f"{metric}_ci"] = f"({low:.4f} - {high:.4f})"
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high

        for threshold, suffix in operating_points:
            row.update(_operating_point_columns(y_split, p_split, threshold, suffix))

        row["screening_threshold"] = round(SCREENING_THRESHOLD, 4)
        row["youden_threshold"] = round(external.youden_threshold, 4)
        row["sens_target_threshold"] = round(external.sensitivity_threshold, 4)
        row["sens_target"] = TARGET_SENSITIVITY
        row["bootstrap_unit"] = external.bootstrap_unit()
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "calibration_uncertainty_by_country.csv", index=False)

    # Table 3 as the manuscript prints it.
    table3 = out[["split", "ece", "ece_ci", "brier", "brier_ci"]].copy()
    table3["ECE (95% CI)"] = [
        f"{value:.4f} {ci}" for value, ci in zip(table3["ece"], table3["ece_ci"])
    ]
    table3["Brier Score (95% CI)"] = [
        f"{value:.4f} {ci}" for value, ci in zip(table3["brier"], table3["brier_ci"])
    ]
    table3 = table3[["split", "ECE (95% CI)", "Brier Score (95% CI)"]]
    table3.to_csv(SAVE_DIR / "table3_calibration_by_cohort.csv", index=False)

    print("\nTable 3 - calibration by cohort (bootstrap 95% CI)")
    print(table3.to_string(index=False))
    print("\nOperating points (Wilson score 95% CI on each rate)")
    for _, row in out.iterrows():
        print(f"  {row['split']} (n={row['n']}, SGA={row['n_pos']})")
        for _, suffix in operating_points:
            print(
                f"    @{suffix:<9} sens={row[f'sensitivity@{suffix}_ci']}  "
                f"spec={row[f'specificity@{suffix}_ci']}"
            )
    print(f"\nSaved to: {SAVE_DIR}")
    return out


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
