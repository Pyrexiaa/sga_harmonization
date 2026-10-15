"""Out-of-fold scoring of every record, complete or not (appendix Table S10).

The held-out test partition of the main analysis contains complete records only,
because only they entered the five-fold split. Here the complete and incomplete
records of each cohort are pooled, folds are reassigned over that union by
stratified group k-fold keyed on the pregnancy identifier, and each fold is held out
in turn while the model and every data-dependent step are fitted on the remaining
four. Every record is therefore scored exactly once by a model that never saw it.

Per held-out fold, fitted on the development folds only and applied unchanged:
within-feature iterative imputation (outcome excluded), the cross-domain imputers,
SMOTENC, standardisation, the Logistic Regression grid search and a Platt calibrator
fitted on the development folds' own out-of-fold predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

from sga.config import (
    COMMON_FEATURES,
    HARMONIZED_SELECTED_FEATURES,
    INDIA,
    LABEL,
    MALAYSIA,
    N_FOLDS_TOTAL,
    SCREENING_THRESHOLD,
    SEED,
)
from sga.data.scaling import scale_feature_test, scale_feature_train
from sga.data.splits import assign_folds
from sga.evaluation.bootstrap import bootstrap_metric_ci
from sga.evaluation.calibration import apply_platt, fit_platt
from sga.evaluation.delong import delong_test_unpaired
from sga.evaluation.metrics import confusion_counts, rate_numerator_denominator
from sga.evaluation.proportions import rate_with_ci
from sga.imputation.apply import impute_df
from sga.imputation.fold_imputers import fit_fold_imputers
from sga.models.estimators import train_lr
from sga.pipeline.dataset import impute_within_feature
from sga.pipeline.harmonized_fold import group_selected_features

COMPLETE = "complete"
INCOMPLETE = "incomplete"
CATEGORICAL = ["gender"]

#: Fewer SGA events than this in a stratum and its metrics are reported as not estimable.
MIN_EVENTS = 5


def build_union(cohort_pair, cohort, id_exists, seed=SEED, n_folds=N_FOLDS_TOTAL):
    """Pool one cohort's complete and incomplete records and reassign folds over them.

    Adds ``record_type`` (complete / incomplete), ``n_imputed_common`` (how many of
    the ten common predictors were missing before imputation), ``country``,
    ``record_uid`` and ``cluster``.
    """
    complete, add_on = (frame.copy() for frame in cohort_pair)
    complete["record_type"] = COMPLETE
    add_on["record_type"] = INCOMPLETE
    union = pd.concat([complete, add_on], ignore_index=True)
    union = union[union[LABEL].notna()].reset_index(drop=True)
    union[LABEL] = union[LABEL].astype(int)
    union["n_imputed_common"] = union[COMMON_FEATURES].isna().sum(axis=1).astype(int)
    union = assign_folds(union.drop(columns="fold"), num_folds=n_folds, id_exist=id_exists, seed=seed)

    prefix = "MY" if cohort == MALAYSIA else "IN"
    union["country"] = cohort
    union["record_uid"] = [f"{prefix}:{i}" for i in range(len(union))]
    union["cluster"] = (
        [f"{prefix}:{i}" for i in union["id"]] if id_exists else union["record_uid"]
    )
    return union


def _feature_columns(frame, selected):
    return list(COMMON_FEATURES) + [f for f in selected if f in frame.columns]


def _within_feature(dev, held, selected, seed):
    columns = _feature_columns(dev, selected)
    dev_part, held_part, _ = impute_within_feature(
        dev[columns + [LABEL]], held[columns + [LABEL]], seed=seed
    )
    dev, held = dev.copy(), held.copy()
    dev[columns] = dev_part[columns].to_numpy()
    held[columns] = held_part[columns].to_numpy()
    for frame in (dev, held):
        frame["gender"] = frame["gender"].round().astype(int)
    return dev, held


def _harmonize(dev_msia, held_msia, dev_india, held_india, selected, seed):
    """Refit the cross-domain imputers on the development rows and fill both cohorts."""
    _, msia_mc, msia_reg, ind_bin, ind_reg = group_selected_features(selected)
    targets = (
        [(f, "multiclass") for f in msia_mc] + [(f, "regression") for f in msia_reg]
        + [(f, "binary") for f in ind_bin] + [(f, "regression") for f in ind_reg]
    )
    imputers = fit_fold_imputers({"malaysia": dev_msia, "india": dev_india}, targets, seed=seed)
    if msia_mc or msia_reg:
        dev_india = impute_df(dev_india, msia_mc + msia_reg, (), msia_mc, msia_reg, imputers=imputers)
        held_india = impute_df(held_india, msia_mc + msia_reg, (), msia_mc, msia_reg, imputers=imputers)
    if ind_bin or ind_reg:
        dev_msia = impute_df(dev_msia, ind_bin + ind_reg, ind_bin, (), ind_reg, imputers=imputers)
        held_msia = impute_df(held_msia, ind_bin + ind_reg, ind_bin, (), ind_reg, imputers=imputers)
    return dev_msia, held_msia, dev_india, held_india


def _fit_scaled_lr(dev, features, seed):
    """SMOTENC on the development rows, standardise, grid-search the LR."""
    continuous = [f for f in features if f not in CATEGORICAL]
    smote = SMOTENC(
        sampling_strategy="auto",
        categorical_features=[features.index(c) for c in CATEGORICAL],
        random_state=seed,
    )
    X, y = smote.fit_resample(dev[features], dev[LABEL].astype(int))
    X = X.copy()
    X[continuous], scale, centre = scale_feature_train(X[continuous], method="std")
    model = train_lr(X, y, seed=seed)

    def score(frame):
        scored = frame[features].copy()
        scored[continuous] = scale_feature_test(scored[continuous], scale, centre, method="std")
        return model.predict_proba(scored)[:, 1]

    return score


def score_held_out_fold(msia_union, india_union, held_out, selected=HARMONIZED_SELECTED_FEATURES, seed=SEED):
    """Fit everything on the other folds and score fold ``held_out`` of both cohorts."""
    dev_msia, held_msia = msia_union[msia_union["fold"] != held_out], msia_union[msia_union["fold"] == held_out]
    dev_india, held_india = india_union[india_union["fold"] != held_out], india_union[india_union["fold"] == held_out]

    dev_msia, held_msia = _within_feature(dev_msia, held_msia, selected, seed)
    dev_india, held_india = _within_feature(dev_india, held_india, selected, seed)
    dev_msia, held_msia, dev_india, held_india = _harmonize(
        dev_msia, held_msia, dev_india, held_india, selected, seed
    )

    features = list(COMMON_FEATURES) + list(selected)
    dev = pd.concat([dev_msia, dev_india], ignore_index=True)
    held = pd.concat([held_msia, held_india], ignore_index=True)

    oof_y, oof_p = [], []
    for inner in sorted(dev["fold"].unique()):
        inner_train, inner_val = dev[dev["fold"] != inner], dev[dev["fold"] == inner]
        if inner_train[LABEL].nunique() < 2 or inner_val.empty:
            continue
        oof_y.append(inner_val[LABEL].to_numpy())
        oof_p.append(_fit_scaled_lr(inner_train, features, seed)(inner_val))
    calibrator = fit_platt(np.concatenate(oof_y), np.concatenate(oof_p))

    p_raw = _fit_scaled_lr(dev, features, seed)(held)
    out = held[["record_uid", "cluster", "country", "record_type", "n_imputed_common", LABEL]].copy()
    out["fold"] = held_out
    out["prob_raw"] = p_raw
    out["prob"] = apply_platt(calibrator, p_raw)
    return out


def score_all_records(msia_union, india_union, selected=HARMONIZED_SELECTED_FEATURES, seed=SEED, verbose=True):
    """Score every record once, each by the model of the fold that held it out."""
    folds = sorted(set(msia_union["fold"]) | set(india_union["fold"]))
    parts = []
    for held_out in folds:
        scored = score_held_out_fold(msia_union, india_union, held_out, selected, seed)
        parts.append(scored)
        if verbose:
            print(f"  held-out fold {held_out}: {len(scored)} records scored")
    predictions = pd.concat(parts, ignore_index=True)
    expected = len(msia_union) + len(india_union)
    if predictions["record_uid"].nunique() != expected or len(predictions) != expected:
        raise AssertionError("every record must be scored exactly once")
    return predictions


def imputation_burden(n_imputed):
    """Stratum label for the number of common predictors imputed (Table S10b)."""
    if n_imputed == 0:
        return "no common predictor imputed"
    if n_imputed == 1:
        return "one predictor imputed"
    if n_imputed == 2:
        return "two predictors imputed"
    return "three or more imputed"


def stratum_metrics(frame, name, threshold=SCREENING_THRESHOLD, n_boot=None, seed=SEED):
    """n, events, cluster-bootstrap AUROC/AUPRC and Wilson sensitivity/specificity."""
    y = frame[LABEL].to_numpy().astype(int)
    p = frame["prob"].to_numpy()
    row = {"stratum": name, "n_scans": int(len(y)), "sga_events": int(y.sum())}
    if row["sga_events"] < MIN_EVENTS or row["sga_events"] == row["n_scans"]:
        row["note"] = f"not estimable: {row['sga_events']} SGA events"
        return row

    kwargs = {"seed": seed, "cluster_ids": frame["cluster"].to_numpy()}
    if n_boot is not None:
        kwargs["n_boot"] = n_boot
    for metric in ("auroc", "auprc"):
        point, low, high = bootstrap_metric_ci(y, p, metric=metric, **kwargs)
        row.update({metric: point, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
    counts = confusion_counts(y, p >= threshold)
    for rate in ("sensitivity", "specificity"):
        interval = rate_with_ci(*rate_numerator_denominator(counts, rate))
        row.update({rate: interval.point, f"{rate}_ci_low": interval.low, f"{rate}_ci_high": interval.high})
    row["note"] = ""
    return row


def _cohort_name(code):
    return "Malaysia" if code == MALAYSIA else "India"


def table_by_record_type(predictions, **kwargs):
    """Table S10a: all records, complete vs incomplete, and each cohort by record type."""
    rows = [stratum_metrics(predictions, "All records", **kwargs)]
    for record_type in (COMPLETE, INCOMPLETE):
        rows.append(stratum_metrics(predictions[predictions["record_type"] == record_type],
                                    f"{record_type.capitalize()} records", **kwargs))
    for cohort in (MALAYSIA, INDIA):
        for record_type in (COMPLETE, INCOMPLETE):
            subset = predictions[(predictions["country"] == cohort) & (predictions["record_type"] == record_type)]
            rows.append(stratum_metrics(subset, f"{_cohort_name(cohort)} ({record_type})", **kwargs))
    return pd.DataFrame(rows)


def table_by_imputation_burden(predictions, **kwargs):
    """Table S10b: records divided by how many common predictors were imputed."""
    burden = predictions["n_imputed_common"].map(imputation_burden)
    rows = [stratum_metrics(predictions, "All records", **kwargs)]
    for label in ("no common predictor imputed", "one predictor imputed",
                  "two predictors imputed", "three or more imputed"):
        rows.append(stratum_metrics(predictions[burden == label], label, **kwargs))
    return pd.DataFrame(rows)


def record_type_by_burden(predictions):
    """Cross-tabulation showing how the two stratifications partition the same records."""
    return pd.crosstab(
        predictions["record_type"],
        predictions["n_imputed_common"].map(imputation_burden),
        margins=True,
    )


def delong_complete_vs_incomplete(predictions):
    """Table S10c: complete vs incomplete records by the unpaired DeLong test."""
    rows = []
    for name, subset in (
        ("Total", predictions),
        ("Malaysia", predictions[predictions["country"] == MALAYSIA]),
        ("India", predictions[predictions["country"] == INDIA]),
    ):
        complete = subset[subset["record_type"] == COMPLETE]
        incomplete = subset[subset["record_type"] == INCOMPLETE]
        auc_c, auc_i, z, p = delong_test_unpaired(
            complete[LABEL], complete["prob"], incomplete[LABEL], incomplete["prob"]
        )
        rows.append(
            {
                "cohort": name,
                "n_complete": len(complete),
                "n_incomplete": len(incomplete),
                "auroc_complete": auc_c,
                "auroc_incomplete": auc_i,
                "difference": auc_c - auc_i,
                "z": z,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)
