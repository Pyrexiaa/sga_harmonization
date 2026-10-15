"""Does amniotic fluid condition add anything on top of ute_ari? (Table 7a decision)

Three feature sets go through the same calibrated external-fold construction as the
main analysis - the ten common predictors alone, plus ute_ari, and plus ute_ari and
af - and are compared on the identical fold-4 rows, per cohort, by paired DeLong
tests and paired bootstrap intervals for the AUROC and AUPRC differences.

Run:
    python -m rebuttals.round3.experiment_R2_2b_af_downstream
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sga.config import ROUND3_DIR, SEED, set_seed
from sga.evaluation.bootstrap import bootstrap_difference_ci, bootstrap_metric_ci
from sga.evaluation.delong import delong_test
from sga.pipeline.external_fold import build_calibrated_external_fold

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND3_DIR / "R2_2b_af_downstream"

ARMS = {
    "common": [],
    "common + ute_ari": ["ute_ari"],
    "common + ute_ari + af": ["ute_ari", "af"],
}
CONTRASTS = [
    ("common + ute_ari", "common"),
    ("common + ute_ari + af", "common + ute_ari"),
]


def run_experiment():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    folds = {name: build_calibrated_external_fold(selected_features=features)
             for name, features in ARMS.items()}
    rows, contrasts = [], []
    for split, _ in folds["common"].splits():
        for name, fold in folds.items():
            arm_mask = dict(fold.splits())[split]
            arm_y, arm_p = fold.y_true[arm_mask], fold.p_calibrated[arm_mask]
            row = {"split": split, "arm": name, "n": int(len(arm_y)), "n_sga": int(arm_y.sum())}
            for metric in ("auroc", "auprc"):
                point, low, high = bootstrap_metric_ci(arm_y, arm_p, metric=metric)
                row.update({metric: point, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
            rows.append(row)

        for with_name, without_name in CONTRASTS:
            with_fold, without_fold = folds[with_name], folds[without_name]
            if not np.array_equal(with_fold.y_true, without_fold.y_true):
                print(f"  [skip] {with_name} vs {without_name}: arms scored different rows")
                continue
            pair_mask = dict(with_fold.splits())[split]
            y_pair = with_fold.y_true[pair_mask]
            p_with = with_fold.p_calibrated[pair_mask]
            p_without = without_fold.p_calibrated[pair_mask]
            _, _, z, p_value = delong_test(y_pair, p_with, p_without)
            contrast = {"split": split, "contrast": f"{with_name} minus {without_name}",
                        "delong_z": z, "delong_p": p_value}
            for metric in ("auroc", "auprc"):
                point, low, high = bootstrap_difference_ci(y_pair, p_with, p_without, metric=metric)
                contrast.update({f"{metric}_difference": point,
                                 f"{metric}_difference_ci_low": low,
                                 f"{metric}_difference_ci_high": high})
            contrasts.append(contrast)

    arms = pd.DataFrame(rows)
    differences = pd.DataFrame(contrasts)
    arms.to_csv(SAVE_DIR / "arm_discrimination.csv", index=False)
    differences.to_csv(SAVE_DIR / "paired_contrasts.csv", index=False)
    print(arms.to_string(index=False))
    print("\n" + differences.to_string(index=False))
    print(f"\nSaved to: {SAVE_DIR}")
    return arms, differences


if __name__ == "__main__":
    run_experiment()
