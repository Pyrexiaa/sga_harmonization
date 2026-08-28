"""The one construction of the external test fold that every reported number uses.

Section 3 of the manuscript reports calibration, operating points, the threshold
sweep, the cluster-aware intervals and the cohort-level fairness analysis from the
SAME model, the SAME calibrator and the SAME fold-4 rows. When each experiment
rebuilt that setup for itself the four could silently drift apart - the 0.50 row of
the threshold sweep stopped matching the sensitivity quoted in the text, and the
fairness section ended up computed on a different partition entirely. Building it
once, here, makes that class of drift impossible.

What this implements, per Methods "Leakage-Safe Cross-Validation and Calibration
Pipeline":

* Folds 0-3 are the development block; fold 4 is the external test partition and is
  never seen during imputer fitting, feature selection, resampling, hyperparameter
  tuning or calibration.
* Platt scaling is fitted by cross-fitting on held-out VALIDATION predictions -
  out-of-fold predictions over the development block - and applied unchanged to the
  fold-4 probabilities. The calibrator therefore never sees a label it is later
  reported against.
* Any alternative operating point is selected on those same development-block
  out-of-fold predictions, never on the test fold.

Why the out-of-fold predictions are built fold by fold rather than with
``cross_val_predict`` over the assembled training matrix. That matrix is SMOTENC
oversampled, so its prevalence is 50% by construction, while fold 4 carries the
observed ~21.6%. A Platt map fitted on the first and applied to the second is
anchored to the wrong base rate: it cannot reduce the expected calibration error,
and a threshold chosen on that scale does not mean the same thing on the test fold
(it was why the Youden point landed far from the ROC corner). Oversampling is
applied to TRAINING rows only, so each development fold's held-out rows are
authentic and sit at the natural prevalence - which is exactly what "held-out
validation predictions" means, and what this module now uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sga.config import (
    EXTERNAL_TEST_FOLD,
    HARMONIZED_SELECTED_FEATURES,
    N_FOLDS_CV,
    SEED,
    set_seed,
)
from sga.evaluation.calibration import apply_platt, fit_platt, platt_cross_fitted
from sga.evaluation.metrics import threshold_for_sensitivity, youden_threshold
from sga.models.estimators import train_lr
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA, prepare_fold

#: Sensitivity floor for the alternative higher-sensitivity operating point.
TARGET_SENSITIVITY = 0.80


@dataclass
class CalibratedExternalFold:
    """The fold-4 evaluation set, calibrated, with everything reported from it."""

    y_true: np.ndarray
    p_raw: np.ndarray
    p_calibrated: np.ndarray
    country_arr: np.ndarray | None
    cluster_ids: np.ndarray | None
    model: object
    calibrator: object
    youden_threshold: float
    sensitivity_threshold: float
    fold_data: object
    development_y: np.ndarray | None = None
    development_p_raw: np.ndarray | None = None
    development_p_calibrated: np.ndarray | None = None
    selected_fold: int | None = None
    validation_auroc: dict | None = None

    @property
    def development_prevalence(self) -> float:
        """SGA prevalence of the held-out validation predictions the calibrator saw.

        This should sit close to the external fold's prevalence. If it drifts
        towards 0.5 the calibrator has been fitted on oversampled rows and neither
        the calibrated probabilities nor any threshold chosen on them transfers.
        """
        if self.development_y is None or len(self.development_y) == 0:
            return float("nan")
        return float(np.mean(self.development_y))

    @property
    def external_prevalence(self) -> float:
        """SGA prevalence of the external test fold."""
        return float(np.mean(self.y_true)) if len(self.y_true) else float("nan")

    @property
    def has_cohorts(self) -> bool:
        """True when the Malaysia/India split of the test rows is usable."""
        return self.country_arr is not None

    def splits(self):
        """``(name, mask)`` for the total cohort and each country, in report order."""
        masks = [("total", np.ones(len(self.y_true), dtype=bool))]
        if self.has_cohorts:
            masks += [
                ("malaysia", self.country_arr == MALAYSIA),
                ("india", self.country_arr == INDIA),
            ]
        return [(name, mask) for name, mask in masks if mask.sum() > 0]

    def composition(self):
        """Per-split n, event count and prevalence - the 807/169 and 46/15 counts."""
        rows = []
        for name, mask in self.splits():
            events = int(self.y_true[mask].sum())
            rows.append(
                {
                    "split": name,
                    "n_scans": int(mask.sum()),
                    "n_sga": events,
                    "prevalence": events / int(mask.sum()),
                    "n_pregnancies": (
                        int(pd.unique(self.cluster_ids[mask]).size)
                        if self.cluster_ids is not None
                        else None
                    ),
                }
            )
        return pd.DataFrame(rows)

    def bootstrap_unit(self):
        """Human-readable description of what the cluster bootstrap resamples."""
        if self.cluster_ids is None:
            return "individual scans (no pregnancy identifier available)"
        return "whole pregnancies (cluster bootstrap)"


def development_fold_models(
    selected_features=HARMONIZED_SELECTED_FEATURES,
    train_source="both",
    n_folds=N_FOLDS_CV,
    seed=SEED,
    verbose=True,
):
    """Fit one model per development fold and score it on that fold's held-out rows.

    For fold ``f`` the model is fitted on the OTHER development folds - oversampled,
    with its own within-feature imputer, cross-domain imputers and scaler, all
    fitted on those training rows - and scored on fold ``f``'s AUTHENTIC held-out
    rows. Concatenating the four gives one prediction per development record from a
    model that never saw it, at the cohort's natural prevalence.

    Fold 4 is absent from these frames entirely (``exclude_external_fold=True``), so
    nothing here can touch the external partition.

    Returns a list of dicts with ``fold``, ``y``, ``p`` and ``validation_auroc``.
    """
    msia_ds, india_ds = load_both_cohorts(exclude_external_fold=True)

    records = []
    for fold in range(n_folds):
        development = prepare_fold(
            msia_ds,
            india_ds,
            fold,
            selected_features=selected_features,
            train_source=train_source,
        )
        model = train_lr(development["train_X"], development["train_Y"], seed=seed)
        y = np.asarray(development["test_Y"]).astype(int)
        p = model.predict_proba(development["test_X"])[:, 1]
        auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        records.append({"fold": fold, "y": y, "p": p, "validation_auroc": float(auroc)})
        if verbose:
            print(
                f"  development fold {fold}: {len(y)} held-out rows, "
                f"{int(y.sum())} SGA, validation AUROC {auroc:.4f}"
            )
    return records


def development_out_of_fold_predictions(**kwargs):
    """Pooled ``(y_true, p_raw)`` over every development fold's held-out rows."""
    records = development_fold_models(**kwargs)
    return (
        np.concatenate([r["y"] for r in records]),
        np.concatenate([r["p"] for r in records]),
    )


def build_calibrated_external_fold(
    selected_features=HARMONIZED_SELECTED_FEATURES,
    train_source="both",
    external_test_fold=EXTERNAL_TEST_FOLD,
    seed=SEED,
    target_sensitivity=TARGET_SENSITIVITY,
    n_development_folds=N_FOLDS_CV,
    verbose=True,
):
    """Score the best development fold-model on the external partition.

    Follows Methods 2.3.4 - "the best model, judging from the highest AUROC based on
    the validation sets, was used to evaluate on the testing data" - rather than
    refitting a fresh model on the pooled development block. The selected fold-model
    carries its own within-feature imputer, cross-domain imputers and scaler onto
    fold 4, so the external rows are transformed exactly as that model's training
    rows were.
    """
    set_seed(seed)

    if verbose:
        print("Fitting development fold-models (folds 0-3) and scoring their held-out rows...")
    records = development_fold_models(
        selected_features=selected_features,
        train_source=train_source,
        n_folds=n_development_folds,
        seed=seed,
        verbose=verbose,
    )
    development_y = np.concatenate([r["y"] for r in records])
    development_p = np.concatenate([r["p"] for r in records])
    validation_auroc = {r["fold"]: r["validation_auroc"] for r in records}

    # One calibrator, fitted on the pooled held-out validation predictions and
    # applied unchanged to fold 4. No fold-4 label enters it.
    calibrator = fit_platt(development_y, development_p)

    # The operating points are chosen on the development block too, but through a
    # CROSS-FITTED calibrator, so no development record is scored by a calibrator
    # its own label helped fit. That is the guarantee the Methods state, and it
    # keeps the threshold search honest rather than optimistic.
    development_calibrated = platt_cross_fitted(development_y, development_p, seed=seed)
    t_youden = youden_threshold(development_y, development_calibrated)
    t_sensitivity = threshold_for_sensitivity(
        development_y, development_calibrated, target=target_sensitivity
    )

    best_fold = max(validation_auroc, key=lambda f: validation_auroc[f])

    # Rebuild the selected fold with fold 4 as its test partition. The training
    # partition, the seeds and every fitted transform are the same as in the loop
    # above, so this reproduces that fold-model rather than creating a new one.
    set_seed(seed)
    msia_dev, india_dev = load_both_cohorts(exclude_external_fold=True)
    msia_full, india_full = load_both_cohorts(exclude_external_fold=False)
    fold_data = prepare_fold(
        msia_dev,
        india_dev,
        best_fold,
        selected_features=selected_features,
        train_source=train_source,
        msia_ds_full=msia_full,
        india_ds_full=india_full,
        external_test_fold=external_test_fold,
    )
    model = train_lr(fold_data["train_X"], fold_data["train_Y"], seed=seed)
    p_raw = model.predict_proba(fold_data["test_X"])[:, 1]
    p_calibrated = apply_platt(calibrator, p_raw)

    result = CalibratedExternalFold(
        y_true=np.asarray(fold_data["test_Y"]).astype(int),
        p_raw=p_raw,
        p_calibrated=p_calibrated,
        country_arr=fold_data["country_arr"],
        cluster_ids=fold_data["cluster_ids"],
        model=model,
        calibrator=calibrator,
        youden_threshold=float(t_youden),
        sensitivity_threshold=float(t_sensitivity),
        fold_data=fold_data,
        development_y=development_y,
        development_p_raw=development_p,
        development_p_calibrated=development_calibrated,
        selected_fold=int(best_fold),
        validation_auroc=validation_auroc,
    )

    if verbose:
        print(
            f"Selected fold-model {best_fold} "
            f"(validation AUROC {validation_auroc[best_fold]:.4f}; "
            + ", ".join(f"fold {f}: {a:.4f}" for f, a in sorted(validation_auroc.items()))
            + ")"
        )
        print(
            f"External fold {external_test_fold}: "
            f"{len(result.y_true)} scans, {int(result.y_true.sum())} SGA"
        )
        print(result.composition().to_string(index=False))
        print(
            f"Calibrator fitted on {len(development_y)} held-out development "
            f"predictions (prevalence {result.development_prevalence:.4f}); "
            f"external fold prevalence {result.external_prevalence:.4f}"
        )
        gap = abs(result.development_prevalence - result.external_prevalence)
        if gap > 0.05:
            print(
                f"  [warn] the two prevalences differ by {gap:.4f}. The Platt map is "
                "anchored to the development base rate, so a large gap means neither "
                "the calibrated probabilities nor any threshold chosen on them "
                "transfers to the test fold."
            )
        print(
            f"Validation-selected operating points: Youden={t_youden:.3f}, "
            f"sensitivity>={target_sensitivity:.2f} -> {t_sensitivity:.3f}"
        )
    return result


__all__ = [
    "TARGET_SENSITIVITY",
    "CalibratedExternalFold",
    "build_calibrated_external_fold",
    "development_fold_models",
    "development_out_of_fold_predictions",
]
