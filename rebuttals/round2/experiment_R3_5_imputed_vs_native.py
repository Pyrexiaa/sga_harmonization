"""Imputed versus natively measured feature contribution.

Run:
    python -m rebuttals.round2.experiment_R3_5_imputed_vs_native
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sga.config import (
    HARMONIZED_SELECTED_FEATURES,
    N_FOLDS_CV,
    ROUND2_DIR,
    SEED,
    set_seed,
)
from sga.evaluation.bootstrap import bootstrap_metric_ci
from sga.models.estimators import train_lr
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA, prepare_fold

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND2_DIR / "R3_5_imputed_vs_native"

#: Evaluation split -> nature of ute_ari/af on that split.
SPLIT_FEATURE_NATURE = [
    ("malaysia", "NATIVE (measured)"),
    ("india", "IMPUTED"),
    ("total", "mixed"),
]


def pooled_country_probs(selected):
    """Pool test labels and probabilities across folds for one feature set."""
    accumulated = {
        "malaysia": {"y": [], "p": []},
        "india": {"y": [], "p": []},
        "total": {"y": [], "p": []},
    }
    for fold in range(N_FOLDS_CV):
        set_seed(SEED)
        msia_ds, india_ds = load_both_cohorts()
        fold_data = prepare_fold(msia_ds, india_ds, fold, selected_features=selected)
        if fold_data["country_arr"] is None:
            continue
        model = train_lr(fold_data["train_X"], fold_data["train_Y"])
        prob = model.predict_proba(fold_data["test_X"])[:, 1]
        country = fold_data["country_arr"]
        accumulated["malaysia"]["y"].append(fold_data["test_Y"][country == MALAYSIA])
        accumulated["malaysia"]["p"].append(prob[country == MALAYSIA])
        accumulated["india"]["y"].append(fold_data["test_Y"][country == INDIA])
        accumulated["india"]["p"].append(prob[country == INDIA])
        accumulated["total"]["y"].append(fold_data["test_Y"])
        accumulated["total"]["p"].append(prob)

    return {
        split: (np.concatenate(parts["y"]), np.concatenate(parts["p"]))
        for split, parts in accumulated.items()
        if parts["y"]
    }


def run_experiment():
    """Compare per-cohort AUPRC with and without the cross-domain features."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print("Building COMMON-only predictions...")
    common = pooled_country_probs(())
    print("Building HARMONIZED (common + ute_ari + af) predictions...")
    harmonized = pooled_country_probs(HARMONIZED_SELECTED_FEATURES)

    rows = []
    for split, nature in SPLIT_FEATURE_NATURE:
        if split not in common or split not in harmonized:
            continue
        y_common, p_common = common[split]
        y_harm, p_harm = harmonized[split]
        auprc_common, common_lo, common_hi = bootstrap_metric_ci(
            y_common, p_common, metric="auprc"
        )
        auprc_harm, harm_lo, harm_hi = bootstrap_metric_ci(y_harm, p_harm, metric="auprc")
        rows.append(
            {
                "cohort": split,
                "feature_nature": nature,
                "auprc_common": auprc_common,
                "auprc_common_ci": f"[{common_lo:.4f}, {common_hi:.4f}]",
                "auprc_harmonized": auprc_harm,
                "auprc_harm_ci": f"[{harm_lo:.4f}, {harm_hi:.4f}]",
                "auprc_gain": auprc_harm - auprc_common,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "imputed_vs_native_auprc.csv", index=False)
    print(out.to_string(index=False))
    print(
        "\nThe Malaysia gain reflects NATIVELY-measured ute_ari/af; the India gain"
        " reflects the SAME features when IMPUTED. Comparing the two deltas"
        " quantifies how much of the AUPRC improvement is imputation-driven."
    )
    print(f"Saved to: {SAVE_DIR}")
    return out


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
