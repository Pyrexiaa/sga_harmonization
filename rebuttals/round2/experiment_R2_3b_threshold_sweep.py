"""Full threshold sweep (appendix Table S4).

Companion to ``experiment_R2_3_calibration_uncertainty``, which reports the
operating points quoted in the Results text. Table S4 instead tabulates
sensitivity and specificity WITH 95% confidence intervals at every cut-off from
0.10 to 0.90 in steps of 0.05, for the Malaysian, Indian and total cohorts.

Method notes that matter for the reported numbers:

* Sensitivity and specificity are proportions, so their intervals are Wilson score
  intervals, per the Statistical Analysis section. At the Indian cut-offs where the
  numerator is 0 or the denominator is 15, a percentile bootstrap returns a
  degenerate interval; Wilson does not. Bootstrap bounds are still emitted in the
  long-form CSV so the two can be compared.
* Sensitivity and specificity are prevalence-INDEPENDENT. PPV and NPV are not, so
  they are given at each cohort's observed prevalence and a reader who needs them
  at their own SGA prevalence pi should convert:
      PPV = sens*pi / (sens*pi + (1-spec)*(1-pi))
      NPV = spec*(1-pi) / (spec*(1-pi) + (1-sens)*pi)
* ECE and Brier are threshold-INDEPENDENT (one value per cohort), so they are not
  swept; they are in ``experiment_R2_3_calibration_uncertainty``.

Model, calibrator and fold-4 rows come from
``sga.pipeline.external_fold.build_calibrated_external_fold``, the same object
every other externally-evaluated number uses, so the 0.50 row of this sweep
reproduces the sensitivity and specificity quoted in the Results exactly.

Run:
    python -m rebuttals.round2.experiment_R2_3b_threshold_sweep
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sga.config import N_BOOTSTRAP, ROUND2_DIR, SEED, THRESHOLD_GRID, set_seed
from sga.evaluation.bootstrap import bootstrap_rate_grid
from sga.evaluation.metrics import confusion_counts, rate_numerator_denominator
from sga.evaluation.proportions import rate_with_ci
from sga.pipeline.external_fold import build_calibrated_external_fold
from sga.reporting.figures import plot_threshold_sweep

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND2_DIR / "R2_3b_threshold_sweep"

#: Rates tabulated in Table S4, in row-block order.
TABLE_S4_RATES = ("sensitivity", "specificity")


def _net_benefit(counts, n, threshold):
    """Decision-curve net benefit ``TP/n - FP/n * pt/(1-pt)`` at this cut-off."""
    if threshold >= 1.0 or n == 0:
        return float("nan")
    odds = threshold / (1 - threshold)
    return counts["tp"] / n - (counts["fp"] / n) * odds


def sweep_cohort(
    y_true, y_prob, thresholds=THRESHOLD_GRID, n_boot=N_BOOTSTRAP, cluster_ids=None
):
    """Every Table S4 quantity for one cohort, with Wilson and bootstrap bounds."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    prevalence = float(y_true.mean()) if n else float("nan")

    # One shared set of resamples for the whole grid and both rates, so the
    # bootstrap bounds down a column are mutually consistent.
    bootstrap_bounds = bootstrap_rate_grid(
        y_true,
        y_prob,
        thresholds,
        rates=TABLE_S4_RATES,
        n_boot=n_boot,
        cluster_ids=cluster_ids,
    )

    rows = []
    for position, threshold in enumerate(thresholds):
        counts = confusion_counts(y_true, (y_prob >= threshold).astype(int))
        row = {
            "threshold": float(threshold),
            "n": n,
            "n_pos": int(y_true.sum()),
            "obs_prevalence": round(prevalence, 4),
            **counts,
            "net_benefit": _net_benefit(counts, n, float(threshold)),
        }
        for rate in ("sensitivity", "specificity", "ppv", "npv"):
            numerator, denominator = rate_numerator_denominator(counts, rate)
            interval = rate_with_ci(numerator, denominator)
            row.update(interval.as_dict(rate))
            row[f"{rate}_n"] = denominator
        # Bootstrap bounds for the two swept rates, reported alongside Wilson.
        for rate in TABLE_S4_RATES:
            low, high = bootstrap_bounds[rate][position]
            row[f"{rate}_boot_ci_low"] = low
            row[f"{rate}_boot_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def build_table_s4(sweep, rate):
    """Table S4's printed layout: thresholds down, cohorts across."""
    wide = sweep.pivot(index="threshold", columns="split", values=f"{rate}_str")
    ordered = [c for c in ("malaysia", "india", "total") if c in wide.columns]
    wide = wide.reindex(columns=ordered)
    wide.columns = [c.capitalize() for c in wide.columns]
    wide.index = [f"{t:.2f}" for t in wide.index]
    wide.index.name = "Threshold"
    return wide


def run_experiment(n_boot=N_BOOTSTRAP):
    """Sweep the decision threshold over every cohort and write Table S4."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    external = build_calibrated_external_fold()
    external.composition().to_csv(SAVE_DIR / "test_fold_composition.csv", index=False)

    frames = []
    for split, mask in external.splits():
        frame = sweep_cohort(
            external.y_true[mask],
            external.p_calibrated[mask],
            thresholds=THRESHOLD_GRID,
            n_boot=n_boot,
            cluster_ids=(
                external.cluster_ids[mask] if external.cluster_ids is not None else None
            ),
        )
        frame.insert(0, "split", split)
        frames.append(frame)

    sweep = pd.concat(frames, ignore_index=True)
    sweep.to_csv(SAVE_DIR / "threshold_sweep_by_country.csv", index=False)

    for rate in TABLE_S4_RATES:
        table = build_table_s4(sweep, rate)
        table.to_csv(SAVE_DIR / f"tableS4_{rate}.csv")
        print(f"\n{rate.upper()} (Wilson score 95% CI)")
        print(table.to_string())

    try:
        plot_threshold_sweep(sweep, SAVE_DIR / "threshold_sweep.png")
    except Exception as error:  # noqa: BLE001 - a missing figure must not lose the CSVs
        print(f"  [plot skipped: {error}]")

    print(
        "\nPPV and NPV are given at each cohort's observed prevalence "
        "(obs_prevalence); convert to another prevalence with Bayes' rule using "
        "the prevalence-independent sensitivity and specificity above."
    )
    print(f"\nSaved to: {SAVE_DIR}")
    return sweep


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
