"""EFW centile as a single-predictor comparator (appendix Table S9).

Runs on the same calibrated external fold, model and calibrator as Tables 3 and S4
(``build_calibrated_external_fold``), using the EFW centile of each held-out scan:

(a) AUROC and AUPRC of the model and of the EFW centile alone, with
    pregnancy-cluster bootstrap 95% CIs;
(b) operating points with Wilson intervals: the EFW < 10th centile rule, the model at
    0.50 and 0.10, and the model at the cut-off that flags the same proportion of
    scans as the rule (equal clinical workload);
(c) a paired DeLong comparison of model and centile, both scored on the same rows;
(d) net benefit of the model, the rule and treat-all across the decision-curve range.

The EFW centile is computed for every scan, extrapolating beyond the tabulated
2.5th-97.5th range rather than dropping those scans.

Run:
    python -m rebuttals.round3.experiment_R3_5_efw_centile_comparator
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sga.config import (
    DCA_THRESHOLD_RANGE,
    DECISION_THRESHOLD,
    EFW_CENTILE_RULE,
    ROUND3_DIR,
    SCREENING_THRESHOLD,
    SEED,
    set_seed,
)
from sga.data.centiles import efw_centile
from sga.evaluation.bootstrap import bootstrap_metric_ci
from sga.evaluation.comparator import centile_risk, net_benefit, operating_point_table
from sga.evaluation.delong import delong_test
from sga.pipeline.external_fold import build_calibrated_external_fold

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND3_DIR / "R3_5_efw_centile_comparator"
RULE_NAME = f"EFW < {EFW_CENTILE_RULE:g}th centile"
MODEL_THRESHOLDS = (DECISION_THRESHOLD, SCREENING_THRESHOLD)


def held_out_predictions(external):
    """One row per held-out scan: label, cohort, cluster, model probability, centile."""
    centiles = efw_centile(external.fold_data.test_raw)
    frame = pd.DataFrame(
        {
            "y_true": external.y_true,
            "country": external.country_arr,
            "cluster": external.cluster_ids,
            "model_probability": external.p_calibrated,
            "ga": external.fold_data.test_raw["ga"].to_numpy(),
            "efw": external.fold_data.test_raw["efw"].to_numpy(),
        }
    )
    frame["efw_centile"] = centiles["efw_centile"].to_numpy()
    frame["centile_status"] = centiles["centile_status"].to_numpy()
    frame["below_rule"] = centiles["below_rule"].to_numpy()
    frame["centile_risk"] = centile_risk(frame["efw_centile"])
    return frame


def discrimination_table(predictions, masks):
    rows = []
    for split, mask in masks:
        subset = predictions[mask]
        for predictor, column in (("model", "model_probability"), ("EFW centile alone", "centile_risk")):
            row = {"split": split, "predictor": predictor,
                   "n_scans": len(subset), "sga_events": int(subset["y_true"].sum())}
            for metric in ("auroc", "auprc"):
                point, low, high = bootstrap_metric_ci(
                    subset["y_true"].to_numpy(), subset[column].fillna(0.0).to_numpy(),
                    metric=metric, cluster_ids=subset["cluster"].to_numpy(),
                )
                row.update({metric: point, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
            rows.append(row)
    return pd.DataFrame(rows)


def operating_points(predictions, masks):
    frames = []
    for split, mask in masks:
        subset = predictions[mask]
        table = operating_point_table(
            subset["y_true"], subset["model_probability"], subset["below_rule"],
            MODEL_THRESHOLDS, RULE_NAME,
        )
        table.insert(0, "split", split)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def paired_delong(predictions, masks):
    rows = []
    for split, mask in masks:
        subset = predictions[mask]
        auc_model, auc_centile, z, p = delong_test(
            subset["y_true"], subset["model_probability"], subset["centile_risk"].fillna(0.0)
        )
        rows.append({"split": split, "auroc_model": auc_model, "auroc_centile": auc_centile,
                     "difference": auc_model - auc_centile, "z": z, "p_value": p})
    return pd.DataFrame(rows)


def net_benefit_table(predictions, masks):
    low, high = DCA_THRESHOLD_RANGE
    thresholds = np.round(np.arange(max(low, 0.01), high + 1e-9, 0.01), 2)
    rows = []
    for split, mask in masks:
        subset = predictions[mask]
        y = subset["y_true"].to_numpy()
        prevalence = float(y.mean())
        for t in thresholds:
            rows.append(
                {
                    "split": split,
                    "threshold": float(t),
                    "model": net_benefit(y, subset["model_probability"] >= t, t),
                    "efw_rule": net_benefit(y, subset["below_rule"], t),
                    "treat_all": prevalence - (1 - prevalence) * t / (1 - t),
                    "treat_none": 0.0,
                }
            )
    return pd.DataFrame(rows)


def run_experiment():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    external = build_calibrated_external_fold()
    predictions = held_out_predictions(external)
    masks = external.splits()

    outside = predictions["centile_status"].value_counts()
    print(f"EFW centile status on the held-out scans:\n{outside.to_string()}")

    tables = {
        "tableS9a_discrimination.csv": discrimination_table(predictions, masks),
        "tableS9b_operating_points.csv": operating_points(predictions, masks),
        "tableS9c_paired_delong.csv": paired_delong(predictions, masks),
        "net_benefit.csv": net_benefit_table(predictions, masks),
    }
    predictions.to_csv(SAVE_DIR / "held_out_predictions.csv", index=False)
    for name, table in tables.items():
        table.to_csv(SAVE_DIR / name, index=False)
        if name != "net_benefit.csv":
            print(f"\n{name}\n{table.to_string(index=False)}")
    print(f"\nSaved to: {SAVE_DIR}")
    return tables


if __name__ == "__main__":
    run_experiment()
