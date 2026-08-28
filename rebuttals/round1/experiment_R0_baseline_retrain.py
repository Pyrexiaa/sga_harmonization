"""Round-1 prerequisite: retrain every model the other round-1 scripts consume.

Run:
    python -m rebuttals.round1.experiment_R0_baseline_retrain
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sga.config import (
    ACCURACY_THRESHOLD,
    EXTERNAL_TEST_FOLD,
    N_BOOTSTRAP,
    N_FOLDS_CV,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.data.scaling import descale_feature
from sga.evaluation.bootstrap import bootstrap_metric_ci
from sga.evaluation.country import append_fold_rows, evaluate_splits, save_country_results
from sga.evaluation.metrics import basic_metrics
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA
from sga.pipeline.model_io import (
    load_trained_model,
    predict_labels_and_proba,
    save_predictions,
    weights_exist,
)
from sga.pipeline.train_unified import (
    build_harmonized_folds,
    train_catboost_unified,
    train_dnn_unified,
    train_sklearn_unified,
)

SAVE_DIR = ROUND1_DIR / "R0_baseline_retrain"
EXTERNAL_PRED_DIR = SAVE_DIR / "external_test_predictions"
# Manual imputation-feature combinations written by
# experiment_R0_baseline_retrain_manual.py; the harmonized arm reported in the
# manuscript is the ``ute_ari`` + ``af`` combination.
SELECTED_IMPUTATION_DIR = SAVE_DIR / "selected_imputation"
HARMONIZED_WEIGHTS_DIR = SELECTED_IMPUTATION_DIR / "sel_ute_ari_af"

FORCE_RETRAIN = False  # set True to retrain even if weights already exist

# The three published DNN configurations (dropout, layer_size, epochs, batch, lr,
# weight_decay, l1_lambda).
DNN_CONFIGS = [
    (0.20, 4, 100, 256, 1e-3, 5e-3, 0),
    (0.20, 4, 100, 256, 1e-3, 1e-3, 0),
    (0.20, 6, 100, 256, 1e-3, 5e-3, 0),
]
ML_MODELS = ["rf", "lr", "svc", "stacking"]
# Positional placeholder tuple; the classical models take their grids from
# `sga.models.hyperparameters`.
ML_HYPERPARAMS = (0.20, 4, 100, 256, 5e-3, 5e-3, 0)

# Malaysia-native features imputed INTO India, by imputation-model type.
IMPUTER_TARGETS = [
    ("malaysia", ["afi", "psv", "ute_ari"]),
    ("malaysia", ["af", "placenta_site"]),
    ("india", ["umb_api", "m_height", "m_weight"]),
    (
        "india",
        [
            "last_preg_sga",
            "last_preg_fgr",
            "last_preg_normal",
            "prev_failed_preg",
            "high_risk_pe",
            "hypertension_0",
            "hypertension_1",
            "diabetes_0",
        ],
    ),
]

# Unified (harmonized, cross-imputation) runs: name -> (family, subdir[, dnn_idx]).
UNIFIED_MODELS = {
    "catboost": ("catboost", f"generalized_catboost_{SEED}"),
    "rf": ("ml", f"generalized_rf_{SEED}"),
    "lr": ("ml", f"generalized_lr_{SEED}"),
    "svc": ("ml", f"generalized_svc_{SEED}"),
    "stacking": ("ml", f"generalized_stacking_{SEED}"),
    "dnn_cfg0": ("dnn", f"generalized_dnn_{SEED}_0", 0),
    "dnn_cfg1": ("dnn", f"generalized_dnn_{SEED}_1", 1),
    "dnn_cfg2": ("dnn", f"generalized_dnn_{SEED}_2", 2),
}

# Single-source baselines (ten COMMON features only):
SINGLE_SOURCE_MODELS = {}
for _source in ("malaysia", "india"):
    SINGLE_SOURCE_MODELS[f"single_source_{_source}_catboost"] = (
        _source, "catboost", f"single_source_{_source}/generalized_catboost_{SEED}", None
    )
    for _model_type in ML_MODELS:
        SINGLE_SOURCE_MODELS[f"single_source_{_source}_{_model_type}"] = (
            _source, "ml", f"single_source_{_source}/generalized_{_model_type}_{SEED}", None
        )
    for _index in range(len(DNN_CONFIGS)):
        SINGLE_SOURCE_MODELS[f"single_source_{_source}_dnn_cfg{_index}"] = (
            _source, "dnn", f"single_source_{_source}/generalized_dnn_cfg{_index}_{SEED}",
            _index,
        )

# The COMMON-FEATURE single-source arms, at their own paths.
SINGLE_SOURCE_COMMON_MODELS = {}
for _source in ("malaysia", "india"):
    SINGLE_SOURCE_COMMON_MODELS[f"single_source_{_source}_catboost_common"] = (
        _source, "catboost",
        f"single_source_{_source}/generalized_catboost_common_{SEED}", None,
    )
    for _model_type in ML_MODELS:
        SINGLE_SOURCE_COMMON_MODELS[f"single_source_{_source}_{_model_type}_common"] = (
            _source, "ml",
            f"single_source_{_source}/generalized_{_model_type}_common_{SEED}", None,
        )
    for _index in range(len(DNN_CONFIGS)):
        SINGLE_SOURCE_COMMON_MODELS[f"single_source_{_source}_dnn_cfg{_index}_common"] = (
            _source, "dnn",
            f"single_source_{_source}/generalized_dnn_cfg{_index}_common_{SEED}", _index,
        )


def _should_skip(download_path, tag):
    """True when a run already has a complete set of fold weights."""
    if not FORCE_RETRAIN and weights_exist(download_path, N_FOLDS_CV):
        print(f"  [skip] {tag}: weights already present in {download_path}/model_weights")
        return True
    return False


def _single_source_dir(source, leaf, common=False):
    """Results root of one single-source run."""
    suffix = "_common" if common else ""
    return str(
        SAVE_DIR / f"single_source_{source}" / f"{leaf}{suffix}_{SEED}" / "malaysia_tri3"
    )


# ── Phase 1: cross-domain imputation models ──────────────────────────────────


def retrain_imputation_models():
    """Refit every per-feature CatBoost imputer used by the harmonization step."""
    from rebuttals.round1.experiment_R2_2_imputation_metrics import (
        train_and_evaluate_feature,
    )

    for country, features in IMPUTER_TARGETS:
        for feature in features:
            print(f"\n{'=' * 60}\nTraining imputation model: {country} -> {feature}\n{'=' * 60}")
            train_and_evaluate_feature(feature, country)


# ── Phase 2: unified (pooled, harmonized) baselines ──────────────────────────


def retrain_unified_catboost():
    """Train the unified CatBoost model."""
    save_dir = SAVE_DIR / f"generalized_catboost_{SEED}"
    if _should_skip(str(save_dir / "malaysia_tri3"), "CatBoost (unified)"):
        return
    msia_ds, india_ds = load_both_cohorts()
    train_catboost_unified(
        msia_ds,
        india_ds,
        download_path=str(save_dir / "malaysia_tri3"),
        smoting=True,
        undersampling=False,
        accuracy_threshold=ACCURACY_THRESHOLD,
    )


def retrain_unified_ml():
    """Train the four unified classical classifiers (RF / LR / SVC / Stacking)."""
    for model_type in ML_MODELS:
        save_dir = SAVE_DIR / f"generalized_{model_type}_{SEED}"
        if _should_skip(str(save_dir / "malaysia_tri3"), f"ML {model_type} (unified)"):
            continue
        msia_ds, india_ds = load_both_cohorts()
        train_sklearn_unified(
            msia_ds,
            india_ds,
            download_path=str(save_dir / "malaysia_tri3"),
            model_type=model_type,
            smoting=True,
            undersampling=False,
            accuracy_threshold=ACCURACY_THRESHOLD,
        )


def retrain_unified_dnn():
    """Train the unified neural network under each published configuration."""
    for index, config in enumerate(DNN_CONFIGS):
        save_dir = SAVE_DIR / f"generalized_dnn_{SEED}_{index}"
        if _should_skip(str(save_dir / "malaysia_tri3"), f"DNN config {index} (unified)"):
            continue
        msia_ds, india_ds = load_both_cohorts()
        train_dnn_unified(
            msia_ds,
            india_ds,
            download_path=str(save_dir / "malaysia_tri3"),
            hyperparameters=config,
            model_size="large",
            smoting=True,
            undersampling=False,
            accuracy_threshold=ACCURACY_THRESHOLD,
        )


# ── Phase 3: single-source baselines ─────────────────────────────────────────


def retrain_single_source_catboost(source, smoting=True):
    """Train the CatBoost baseline on one cohort only (full harmonized space)."""
    print(f"\n--- [single-source={source}] CatBoost ---")
    save_dir = _single_source_dir(source, "generalized_catboost")
    if _should_skip(save_dir, f"[ss={source}] CatBoost"):
        return
    msia_ds, india_ds = load_both_cohorts()
    try:
        train_catboost_unified(
            msia_ds,
            india_ds,
            download_path=save_dir,
            smoting=smoting,
            accuracy_threshold=ACCURACY_THRESHOLD,
            train_source=source,
        )
    except Exception as error:  # noqa: BLE001 - one failing arm must not stop the sweep
        print(f"  [single-source={source}] CatBoost FAILED: {error}")


def retrain_single_source_ml(source, smoting=True):
    """Train the four classical baselines on one cohort only."""
    for model_type in ML_MODELS:
        print(f"\n--- [single-source={source}] ML: {model_type} ---")
        save_dir = _single_source_dir(source, f"generalized_{model_type}")
        if _should_skip(save_dir, f"[ss={source}] ML {model_type}"):
            continue
        msia_ds, india_ds = load_both_cohorts()
        try:
            train_sklearn_unified(
                msia_ds,
                india_ds,
                download_path=save_dir,
                model_type=model_type,
                smoting=smoting,
                accuracy_threshold=ACCURACY_THRESHOLD,
                train_source=source,
            )
        except Exception as error:  # noqa: BLE001
            print(f"  [single-source={source}] ML {model_type} FAILED: {error}")


def retrain_single_source_dnn(source, smoting=True):
    """Train the neural baselines on one cohort only, one run per configuration."""
    for index, config in enumerate(DNN_CONFIGS):
        print(f"\n--- [single-source={source}] DNN config {index}: {config} ---")
        save_dir = _single_source_dir(source, f"generalized_dnn_cfg{index}")
        if _should_skip(save_dir, f"[ss={source}] DNN config {index}"):
            continue
        msia_ds, india_ds = load_both_cohorts()
        try:
            train_dnn_unified(
                msia_ds,
                india_ds,
                download_path=save_dir,
                hyperparameters=config,
                smoting=smoting,
                accuracy_threshold=ACCURACY_THRESHOLD,
                pytorch_balanced_sampling=True,
                train_source=source,
            )
        except Exception as error:  # noqa: BLE001
            print(f"  [single-source={source}] DNN config {index} FAILED: {error}")


def retrain_single_source_all():
    """Train every single-source arm for both cohorts."""
    for source in ("malaysia", "india"):
        print(f"\nPHASE 3 -- SINGLE SOURCE: train on {source.upper()} only")
        retrain_single_source_catboost(source)
        retrain_single_source_ml(source)
        retrain_single_source_dnn(source)


def _retrain_single_source_common_one(family, source, leaf, config=None,
                                      model_type=None, smoting=True):
    """Train ONE single-source model on the ten COMMON features only."""
    save_dir = _single_source_dir(source, leaf, common=True)
    msia_ds, india_ds = load_both_cohorts()
    common = {
        "download_path": save_dir,
        "smoting": smoting,
        "accuracy_threshold": ACCURACY_THRESHOLD,
        "train_source": source,
        "selected_features": [],
    }
    try:
        if family == "catboost":
            train_catboost_unified(msia_ds, india_ds, **common)
        elif family == "ml":
            train_sklearn_unified(msia_ds, india_ds, model_type=model_type, **common)
        elif family == "dnn":
            train_dnn_unified(
                msia_ds, india_ds, hyperparameters=config,
                pytorch_balanced_sampling=True, **common
            )
        else:
            raise ValueError(f"unknown family {family!r}")
    except Exception as error:  # noqa: BLE001
        print(f"  [ss={source}] {leaf} (common) FAILED: {error}")


def retrain_single_source_common(source, families=("catboost", "ml", "dnn"), smoting=True):
    """Retrain the single-source baselines on the ten common features."""
    print(f"\nSINGLE-SOURCE retrain on COMMON features -- train_source={source}")
    if "catboost" in families:
        print(f"--- [ss={source}] CatBoost (common) ---")
        _retrain_single_source_common_one(
            "catboost", source, "generalized_catboost", smoting=smoting
        )
    if "ml" in families:
        for model_type in ML_MODELS:
            print(f"--- [ss={source}] ML {model_type} (common) ---")
            _retrain_single_source_common_one(
                "ml", source, f"generalized_{model_type}",
                model_type=model_type, smoting=smoting,
            )
    if "dnn" in families:
        for index, config in enumerate(DNN_CONFIGS):
            print(f"--- [ss={source}] DNN cfg{index} (common) ---")
            _retrain_single_source_common_one(
                "dnn", source, f"generalized_dnn_cfg{index}",
                config=config, smoting=smoting,
            )


def retrain_single_source_lr_common(source, smoting=True):
    """Retrain only the single-source Logistic Regression on the common features."""
    _retrain_single_source_common_one(
        "ml", source, "generalized_lr", model_type="lr", smoting=smoting
    )


def retrain_single_source_common_all(families=("catboost", "ml", "dnn")):
    """Retrain both single-source baselines on the ten common features."""
    for source in ("malaysia", "india"):
        retrain_single_source_common(source, families=families)


# ── External-test (fold 4) inference from the saved weights ──────────────────


def _clinical_units(prepared):
    """Undo standardisation so the saved prediction CSVs are in clinical units."""
    frame = prepared.test_df[prepared.features].copy()
    continuous = [c for c in prepared.continuous_features if c in frame.columns]
    if continuous:
        frame[continuous] = descale_feature(
            frame[continuous], prepared.std_or_min, prepared.mean_or_max
        )
    return frame.reset_index(drop=True)


def build_external_folds(selected_features=None, train_source=None,
                         accuracy_threshold=ACCURACY_THRESHOLD, seed=SEED):
    """Rebuild the fold-4 external test set once per fold-model."""
    set_seed(seed)
    msia_ds, india_ds = load_both_cohorts(exclude_external_fold=True)
    msia_full, india_full = load_both_cohorts(exclude_external_fold=False)

    folds = {}
    for prepared in build_harmonized_folds(
        msia_ds,
        india_ds,
        num_of_folds=N_FOLDS_CV,
        accuracy_threshold=accuracy_threshold,
        train_source=train_source,
        selected_features=selected_features,
        msia_ds_full=msia_full,
        india_ds_full=india_full,
        external_test_fold=EXTERNAL_TEST_FOLD,
    ):
        folds[prepared.fold] = (
            _clinical_units(prepared),
            prepared.test_df[prepared.features].reset_index(drop=True),
            prepared.test_Y.to_numpy(dtype=int),
            prepared.country_arr,
            prepared.features,
        )
    return folds


def build_development_folds(selected_features=None, train_source=None,
                            accuracy_threshold=ACCURACY_THRESHOLD, seed=SEED):
    """Rebuild each development fold's own held-out partition (folds 0-3)."""
    set_seed(seed)
    msia_ds, india_ds = load_both_cohorts(exclude_external_fold=True)

    folds = {}
    for prepared in build_harmonized_folds(
        msia_ds,
        india_ds,
        num_of_folds=N_FOLDS_CV,
        accuracy_threshold=accuracy_threshold,
        train_source=train_source,
        selected_features=selected_features,
    ):
        folds[prepared.fold] = (
            prepared.test_df[prepared.features].reset_index(drop=True),
            prepared.test_Y.to_numpy(dtype=int),
            prepared.country_arr,
            prepared.features,
        )
    return folds


def run_external_inference():
    """Score every saved unified fold-model on the fold-4 external test set."""
    print(f"\nEXTERNAL-TEST INFERENCE (fold {EXTERNAL_TEST_FOLD}) -- load weights, predict, save CSVs")
    EXTERNAL_PRED_DIR.mkdir(parents=True, exist_ok=True)

    harmonized = build_external_folds()

    for name, spec in UNIFIED_MODELS.items():
        family, subdir = spec[0], spec[1]
        dnn_config = DNN_CONFIGS[spec[2]] if len(spec) > 2 else None
        print(f"\n--- {name} ({family}) ---")
        probability_stack, raw_df, y_true, country = [], None, None, None
        for fold in range(N_FOLDS_CV):
            raw_df, test_X, y_true, country, features = harmonized[fold]
            model = load_trained_model(
                family,
                str(SAVE_DIR / subdir / "malaysia_tri3"),
                fold,
                n_features=len(features),
                dnn_config=dnn_config,
            )
            if model is None:
                print(f"    [fold {fold}] weights missing -- skipped")
                continue
            y_pred, y_prob = predict_labels_and_proba(model, family, test_X)
            save_predictions(
                str(EXTERNAL_PRED_DIR / name / f"fold_{fold}.csv"),
                raw_df, y_true, y_pred, y_prob, country,
            )
            probability_stack.append(y_prob)
        if probability_stack and raw_df is not None:
            ensemble = np.mean(np.vstack(probability_stack), axis=0)
            save_predictions(
                str(EXTERNAL_PRED_DIR / name / "ensemble.csv"),
                raw_df, y_true, (ensemble >= 0.5).astype(int), ensemble, country,
            )

    # The common-features LR lives at a flat path and uses the ten common features only
    # (no cross-imputation).
    print("\n--- lr_common_features (common-features LR, flat path) ---")
    common = build_external_folds(selected_features=[])
    probability_stack, raw_df, y_true, country = [], None, None, None
    for fold in range(N_FOLDS_CV):
        raw_df, test_X, y_true, country, features = common[fold]
        model = load_trained_model(
            "ml", str(SAVE_DIR / "lr_common_features"), fold, weights_subdir=""
        )
        if model is None:
            print(f"    [fold {fold}] weights missing at lr_common_features -- skipped")
            continue
        y_pred, y_prob = predict_labels_and_proba(model, "ml", test_X)
        save_predictions(
            str(EXTERNAL_PRED_DIR / "lr_common_features" / f"fold_{fold}.csv"),
            raw_df, y_true, y_pred, y_prob, country,
        )
        probability_stack.append(y_prob)
    if probability_stack and raw_df is not None:
        ensemble = np.mean(np.vstack(probability_stack), axis=0)
        save_predictions(
            str(EXTERNAL_PRED_DIR / "lr_common_features" / "ensemble.csv"),
            raw_df, y_true, (ensemble >= 0.5).astype(int), ensemble, country,
        )

    print(f"\nExternal-test prediction CSVs saved under: {EXTERNAL_PRED_DIR}")


def _round_ci(value):
    """Round floats to 4 dp, leaving NaN and integers untouched."""
    return round(value, 4) if isinstance(value, float) and not np.isnan(value) else value


def _auc_ci_by_split(y_true, y_prob, country_arr, n_boot=N_BOOTSTRAP):
    """Per-split AUROC with a bootstrap 95% CI, plus AUPRC and Brier."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    country_arr = np.asarray(country_arr)

    masks = {
        "total": np.ones(len(country_arr), dtype=bool),
        "malaysia": country_arr == MALAYSIA,
        "india": country_arr == INDIA,
    }
    out = {}
    for split, mask in masks.items():
        y, p = y_true[mask], y_prob[mask]
        metrics = basic_metrics(y, p)
        _, low, high = bootstrap_metric_ci(y, p, metric="auroc", n_boot=n_boot)
        out[split] = {
            "n": metrics["n"],
            "n_pos": metrics["n_pos"],
            "auroc": metrics["auroc"],
            "auroc_ci_low": low,
            "auroc_ci_high": high,
            "auprc": metrics["auprc"],
            "brier": metrics["brier"],
        }
    return out


def run_single_source_inference(models=None):
    """Score the single-source common-feature baselines on the external fold."""
    print(f"\nSINGLE-SOURCE external inference (10 common features, fold {EXTERNAL_TEST_FOLD})")
    EXTERNAL_PRED_DIR.mkdir(parents=True, exist_ok=True)

    items = (
        list(SINGLE_SOURCE_MODELS.items())
        if models is None
        else [(key, SINGLE_SOURCE_MODELS[key]) for key in models]
    )

    # The fold-4 common-feature test set depends only on the source cohort.
    fold_cache = {}

    def folds_for(source):
        if source not in fold_cache:
            fold_cache[source] = build_external_folds(
                selected_features=[], train_source=source
            )
        return fold_cache[source]

    metric_rows, ci_rows = [], []

    def record_ci(name, source, fold_key, y_true, y_prob, country_arr):
        """Attach the AUROC CIs to the metric rows and to a standalone table."""
        cis = _auc_ci_by_split(y_true, y_prob, country_arr)
        for row in metric_rows[len(metric_rows) - len(cis):]:
            split = str(row.get("eval_split", "")).strip().lower()
            if split in cis:
                row["auroc_ci_low"] = _round_ci(cis[split]["auroc_ci_low"])
                row["auroc_ci_high"] = _round_ci(cis[split]["auroc_ci_high"])
        for split, values in cis.items():
            ci_rows.append({
                "model": name, "train_source": source, "fold": fold_key,
                "eval_split": split, "n": values["n"], "n_pos": values["n_pos"],
                "auroc": _round_ci(values["auroc"]),
                "auroc_ci_low": _round_ci(values["auroc_ci_low"]),
                "auroc_ci_high": _round_ci(values["auroc_ci_high"]),
                "auprc": _round_ci(values["auprc"]),
                "brier": _round_ci(values["brier"]),
            })

    for name, (source, family, subdir, dnn_index) in items:
        print(f"\n--- {name}  (train_source={source}, {family}) ---")
        harmonized = folds_for(source)
        dnn_config = DNN_CONFIGS[dnn_index] if dnn_index is not None else None
        probability_stack, raw_df, y_true, country = [], None, None, None
        for fold in range(N_FOLDS_CV):
            raw_df, test_X, y_true, country, features = harmonized[fold]
            model = load_trained_model(
                family,
                str(SAVE_DIR / subdir / "malaysia_tri3"),
                fold,
                n_features=len(features),
                dnn_config=dnn_config,
            )
            if model is None:
                print(f"    [fold {fold}] weights missing under {subdir} -- skipped")
                continue
            y_pred, y_prob = predict_labels_and_proba(model, family, test_X)
            save_predictions(
                str(EXTERNAL_PRED_DIR / name / f"fold_{fold}.csv"),
                raw_df, y_true, y_pred, y_prob, country,
            )
            append_fold_rows(
                metric_rows,
                evaluate_splits(y_true, y_pred, y_prob, country),
                name, fold, extra={"train_source": source},
            )
            record_ci(name, source, fold, y_true, y_prob, country)
            probability_stack.append(y_prob)

        if probability_stack and raw_df is not None:
            ensemble = np.mean(np.vstack(probability_stack), axis=0)
            ensemble_pred = (ensemble >= 0.5).astype(int)
            save_predictions(
                str(EXTERNAL_PRED_DIR / name / "ensemble.csv"),
                raw_df, y_true, ensemble_pred, ensemble, country,
            )
            append_fold_rows(
                metric_rows,
                evaluate_splits(y_true, ensemble_pred, ensemble, country),
                name, "ensemble", extra={"train_source": source},
            )
            record_ci(name, source, "ensemble", y_true, ensemble, country)

    if metric_rows:
        save_country_results(
            metric_rows, str(EXTERNAL_PRED_DIR),
            prefix="single_source_external_metrics",
            group_cols=("model", "train_source", "eval_split"),
        )
    if ci_rows:
        ci_path = EXTERNAL_PRED_DIR / "single_source_auroc_ci_by_fold.csv"
        pd.DataFrame(ci_rows).to_csv(ci_path, index=False)
        print(f"    saved {ci_path}  ({len(ci_rows)} rows)")

    print(f"\nSingle-source external predictions + metrics saved under: {EXTERNAL_PRED_DIR}")
    return metric_rows


#: DNN configuration used for the cross-domain arm of Figure 3. The single-source DNN
#: weights are stored per configuration, so Figure 3 must name one of them.
FIGURE3_DNN_CONFIG_INDEX = 0


def run_single_source_common_inference(models=None):
    """Score the COMMON-FEATURE single-source arms -- the cross-domain arm of Figure 3.

    ``run_single_source_inference`` scores ``SINGLE_SOURCE_MODELS`` (the full
    cross-imputed feature space) and pools every model's rows into one CSV under
    ``external_test_predictions/``. Figure 3's third arm is a different thing: models
    trained on ONE cohort using only the ten common features, then evaluated on the
    OTHER cohort. ``scripts/05a_figure3_auroc_comparison.py`` looks for those results
    inside each run's own directory, under the same
    ``external_fold<N>_per_fold_results.csv`` name the 04* testers write, so this
    function writes them there rather than into the shared pool.
    """
    print(
        f"\nSINGLE-SOURCE COMMON-FEATURE external inference "
        f"(fold {EXTERNAL_TEST_FOLD}) -- Figure 3 cross-domain arm"
    )
    items = (
        list(SINGLE_SOURCE_COMMON_MODELS.items())
        if models is None
        else [(key, SINGLE_SOURCE_COMMON_MODELS[key]) for key in models]
    )

    fold_cache = {}

    def folds_for(source):
        if source not in fold_cache:
            fold_cache[source] = build_external_folds(
                selected_features=[], train_source=source
            )
        return fold_cache[source]

    written = []
    for name, (source, family, subdir, dnn_index) in items:
        run_dir = SAVE_DIR / subdir / "malaysia_tri3"
        print(f"\n--- {name}  (train_source={source}, {family}) ---")
        harmonized = folds_for(source)
        dnn_config = DNN_CONFIGS[dnn_index] if dnn_index is not None else None

        rows = []
        for fold in range(N_FOLDS_CV):
            _, test_X, y_true, country, features = harmonized[fold]
            model = load_trained_model(
                family, str(run_dir), fold,
                n_features=len(features), dnn_config=dnn_config,
            )
            if model is None:
                print(f"    [fold {fold}] weights missing under {subdir} -- skipped")
                continue
            y_pred, y_prob = predict_labels_and_proba(model, family, test_X)
            append_fold_rows(
                rows,
                evaluate_splits(y_true, y_pred, y_prob, country),
                name, fold, extra={"train_source": source,
                                   "eval_fold": EXTERNAL_TEST_FOLD},
            )

        if not rows:
            print(
                f"    no weights for {name}; run retrain_single_source_common_all() "
                "before building Figure 3."
            )
            continue
        save_country_results(
            rows, str(run_dir), prefix=f"external_fold{EXTERNAL_TEST_FOLD}",
            model_name=name, group_cols=("model", "train_source", "eval_split"),
        )
        written.append(str(run_dir))

    print(f"\nWrote cross-domain results for {len(written)} run(s).")
    return written


def run_experiment():
    """Run the external-test inference over the saved R0 weights."""
    run_external_inference()
    # Figure 3's cross-domain arm. Skips cleanly when the common-feature
    # single-source weights have not been trained yet.
    run_single_source_common_inference()


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
