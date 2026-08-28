"""Pregnancy-cluster-aware statistical inference.

Produces the manuscript's "Pregnancy-Cluster-Aware Analysis" paragraph:

* the unique-pregnancy count, mean and maximum scans per pregnancy in the
  Malaysian cohort (each Indian record is a single pregnancy);
* AUROC on the external test fold with a 95% CI from a bootstrap that resamples
  whole PREGNANCIES rather than individual scans, so the correlation between
  repeated scans of the same pregnancy is carried into the interval;
* a sensitivity analysis restricting each pregnancy to a single index scan.

The cluster key is carried through the fold builder on a marker column and popped
next to ``country_arr`` (``sga/pipeline/harmonized_fold.py``), so it survives the
physiological filter. Recovering it afterwards by position would misalign every
interval whenever a test row was dropped.

AUROC is invariant under the monotone Platt map, so calibrating first changes
nothing here; the calibrated probabilities are used anyway so that every fold-4
number in the paper comes from one object.

Run:
    python -m rebuttals.round2.experiment_R2_4_cluster_inference
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sga.config import N_BOOTSTRAP, ROUND2_DIR, SEED, set_seed
from sga.evaluation.bootstrap import bootstrap_metric_ci
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.external_fold import build_calibrated_external_fold

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND2_DIR / "R2_4_cluster_inference"

#: Metrics reported per cohort under the cluster bootstrap.
CLUSTER_METRICS = ("auroc", "auprc")


def describe_pregnancies():
    """Unique pregnancies and the scans-per-pregnancy distribution."""
    (msia, msia_add), (india, _) = load_both_cohorts(exclude_external_fold=False)
    if "id" not in msia.columns:
        raise SystemExit(
            "The Malaysian cohort has no `id` column, so pregnancies cannot be "
            "counted. Re-run scripts/01a_prepare_malaysia.py."
        )

    ids = pd.concat([msia["id"], msia_add["id"]])
    counts = ids.value_counts()
    descriptives = {
        "malaysia_scans": int(len(msia) + len(msia_add)),
        "malaysia_unique_pregnancies": int(counts.shape[0]),
        "malaysia_scans_per_pregnancy_mean": float(counts.mean()),
        "malaysia_scans_per_pregnancy_max": int(counts.max()),
        "india_records_each_one_pregnancy": int(len(india)),
    }
    return descriptives, counts


def _index_scan_rows(cluster_ids):
    """Row indices of the first scan of each pregnancy, in original order."""
    seen = set()
    keep = []
    for row, cluster in enumerate(cluster_ids):
        if cluster not in seen:
            seen.add(cluster)
            keep.append(row)
    return np.asarray(keep, dtype=int)


def run_experiment(n_boot=N_BOOTSTRAP):
    """Run the descriptive, cluster-bootstrap and index-scan analyses."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    descriptives, counts = describe_pregnancies()
    pd.Series(descriptives).to_csv(SAVE_DIR / "pregnancy_descriptives.csv")
    counts.rename("n_scans").to_csv(SAVE_DIR / "malaysia_scans_per_pregnancy.csv")
    print("Pregnancy descriptives:")
    for key, value in descriptives.items():
        print(f"  {key}: {value}")

    external = build_calibrated_external_fold()
    external.composition().to_csv(SAVE_DIR / "test_fold_composition.csv", index=False)
    if external.cluster_ids is None:
        raise SystemExit(
            "No pregnancy identifier reached the external fold, so the cluster "
            "bootstrap cannot be run as the manuscript describes it."
        )

    rows = []
    for split, mask in external.splits():
        clusters = external.cluster_ids[mask]
        row = {
            "split": split,
            "n_scans": int(mask.sum()),
            "n_pregnancies": int(pd.unique(clusters).size),
            "n_sga": int(external.y_true[mask].sum()),
        }
        for metric in CLUSTER_METRICS:
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
            row[f"{metric}_cluster_ci"] = f"{point:.4f} ({low:.4f} - {high:.4f})"
        rows.append(row)

    cluster_df = pd.DataFrame(rows)
    cluster_df.to_csv(SAVE_DIR / "cluster_bootstrap_auroc.csv", index=False)

    # Sensitivity analysis: one index scan per pregnancy, i.i.d. bootstrap (each
    # retained row is now its own pregnancy, so there is nothing left to cluster).
    keep = _index_scan_rows(external.cluster_ids)
    index_rows = []
    for split, mask in external.splits():
        selected = keep[mask[keep]]
        y_selected = external.y_true[selected]
        if len(selected) == 0 or len(np.unique(y_selected)) < 2:
            continue
        point, low, high = bootstrap_metric_ci(
            y_selected, external.p_calibrated[selected], metric="auroc", n_boot=n_boot
        )
        index_rows.append(
            {
                "split": split,
                "n_pregnancies": int(len(selected)),
                "n_sga": int(y_selected.sum()),
                "auroc_index_scan": point,
                "ci": f"{point:.4f} ({low:.4f} - {high:.4f})",
            }
        )
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(SAVE_DIR / "index_scan_sensitivity.csv", index=False)

    print(f"\nCluster bootstrap ({n_boot} iterations, unit = pregnancy):")
    print(
        cluster_df[
            ["split", "n_scans", "n_pregnancies", "n_sga", "auroc_cluster_ci"]
        ].to_string(index=False)
    )
    print("\nIndex-scan sensitivity (one scan per pregnancy):")
    print(index_df.to_string(index=False))
    print(f"\nSaved to: {SAVE_DIR}")
    return cluster_df, index_df


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
