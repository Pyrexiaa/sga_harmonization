"""Imputation-strategy ablation (manuscript Table 7).

    Run experiment_R0_baseline_retrain.py first (the pretrained CatBoost
    imputers used by strategies D and F-J).

Run:
    python -m rebuttals.round1.experiment_R2_5_imputation_ablation
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from sga.config import (
    ACCURACY_THRESHOLD,
    CATEGORICAL_FEATURES,
    CONSTANT_ZERO_FEATURES,
    LABEL,
    N_FOLDS_CV,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.data.cleaning import remove_illogical_values
from sga.evaluation.country import evaluate_splits, summarize
from sga.evaluation.metrics import EVAL_SPLITS, full_metrics
from sga.models.estimators import train_lr
from sga.pipeline.dataset import (
    cast_common_types,
    load_both_cohorts,
    process_raw_train_and_test_df,
    scale_sample_train_and_test_df,
    separate_df_and_df_add_on,
)
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA, prepare_fold
from sga.pipeline.train_unified import select_cross_domain_features

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND1_DIR / "R2_5_imputation_ablation"
WEIGHTS_DIR = SAVE_DIR / "weights"

# Manual-selection arms: keep the common features and CatBoost-impute only these.
SELECTED_IMPUTE_STRATEGIES = {
    "F_impute_ute_ari": ["ute_ari"],
    "G_impute_ute_ari_af": ["ute_ari", "af"],
    "H_impute_ute_ari_af_last_preg_fgr": ["ute_ari", "af", "last_preg_fgr"],
    "I_impute_ute_ari_af_last_preg_normal": ["ute_ari", "af", "last_preg_normal"],
    "J_impute_ute_ari_af_last_preg_fgr_last_preg_normal":
        ["ute_ari", "af", "last_preg_fgr", "last_preg_normal"],
}


def fit_or_load_and_eval(strategy_name, fold, train_X, train_Y, test_X, test_Y,
                         country_arr, all_results):
    """Fit (or reload) the arm's Logistic Regression and score it per cohort."""
    weights_dir = WEIGHTS_DIR / strategy_name
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"model_{fold}.pkl"

    if weights_path.exists():
        print(f"      [no-retrain] loading existing weights: {weights_path}")
        with open(weights_path, "rb") as handle:
            model = pickle.load(handle)
    else:
        model = train_lr(train_X, train_Y)
        with open(weights_path, "wb") as handle:
            pickle.dump(model, handle)
        print(f"      [saved] {weights_path}")

    y_prob = model.predict_proba(test_X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = np.asarray(test_Y)

    if country_arr is not None and len(country_arr) == len(y_true):
        split_metrics = evaluate_splits(y_true, y_pred, y_prob, country_arr)
    else:
        if country_arr is not None:
            print(f"      [warn] country array ({len(country_arr)}) != test rows "
                  f"({len(y_true)}); reporting TOTAL only.")
        split_metrics = {"total": full_metrics(y_true, y_pred, y_prob)}

    n_features = train_X.shape[1]
    for split in EVAL_SPLITS:
        metrics = split_metrics.get(split)
        if not metrics:
            continue
        row = {"strategy": strategy_name, "fold": fold, "eval_split": split,
               "num_features": n_features}
        row.update(metrics)
        all_results.append(row)

    total = split_metrics["total"]
    print(
        f"      #Feat={n_features}  [total] BAcc={total['balanced_accuracy']:.4f} "
        f"AUC={total['roc_auc']:.4f} AUPRC={total['auprc']:.4f} "
        f"Brier={total['brier_score']:.4f} ECE={total['ece']:.4f}"
    )
    return split_metrics


def _run_prepared_folds(msia_ds, india_ds, num_of_folds, all_results, strategy_name,
                        selected_features):
    """Score one arm whose folds are built by the shared harmonized pipeline."""
    for fold in range(num_of_folds):
        print(f"\n    Fold {fold}:")
        prepared = prepare_fold(msia_ds, india_ds, fold, selected_features=selected_features)
        fit_or_load_and_eval(
            strategy_name, fold, prepared.train_X, prepared.train_Y,
            prepared.test_X, prepared.test_Y, prepared.country_arr, all_results,
        )


def run_no_imputation(msia_ds, india_ds, num_of_folds, all_results,
                      strategy_name="A_no_imputation"):
    """Strategy A: the ten common features, no cross-domain reconstruction."""
    print("\n  Preparing no-imputation pipeline (common features only)...")
    _run_prepared_folds(msia_ds, india_ds, num_of_folds, all_results, strategy_name,
                        selected_features=())


def run_catboost_imputation(msia_ds, india_ds, num_of_folds, all_results,
                            strategy_name="D_default_catboost_imputation"):
    """Strategy D: the paper's CatBoost cross-imputation at the quality threshold."""
    print(f"\n  Preparing DEFAULT CatBoost-imputation pipeline ({strategy_name})...")
    selection = select_cross_domain_features(ACCURACY_THRESHOLD)
    _run_prepared_folds(
        msia_ds, india_ds, num_of_folds, all_results, strategy_name,
        selected_features=selection.msia_impute + selection.india_impute,
    )


def run_selected_catboost_imputation(msia_ds, india_ds, num_of_folds, all_results,
                                     strategy_name, selected_features):
    """Strategies F-J: common features plus a manually chosen imputed subset."""
    print(f"\n  Preparing SELECTED CatBoost-imputation ({strategy_name}): "
          f"common + impute {selected_features}")
    _run_prepared_folds(msia_ds, india_ds, num_of_folds, all_results, strategy_name,
                        selected_features=selected_features)


def _fold_frames(msia_df, msia_add_on, india_df, india_add_on, fold):
    """Fold-wise train/test frames for both cohorts, dtypes normalised."""
    msia_train, msia_test = process_raw_train_and_test_df(
        msia_df, msia_add_on, fold, id_exists=True)
    india_train, india_test = process_raw_train_and_test_df(
        india_df, india_add_on, fold, id_exists=False)
    for frame in (msia_train, india_train, msia_test, india_test):
        cast_common_types(frame)
    return msia_train, msia_test, india_train, india_test


def _finalize_and_eval(msia_train, msia_test, india_train, india_test,
                       categorical_candidates, strategy_name, fold, all_results):
    """Pool, resample, scale and score one non-harmonized ablation fold."""
    common_cols = sorted(set(msia_train.columns) & set(india_train.columns))
    msia_train = msia_train[common_cols]
    msia_test = msia_test[[c for c in common_cols if c in msia_test.columns]]
    india_train = india_train[common_cols]
    india_test = india_test[[c for c in common_cols if c in india_test.columns]]

    n_msia_test, n_india_test = len(msia_test), len(india_test)
    train_df = pd.concat([msia_train, india_train], axis=0)
    train_df[LABEL] = train_df[LABEL].astype(int)
    test_df = pd.concat([msia_test, india_test], axis=0)  # Malaysia first
    test_df[LABEL] = test_df[LABEL].astype(int)

    categorical = [c for c in categorical_candidates if c in train_df.columns]
    continuous = [c for c in train_df.columns if c not in categorical and c != LABEL]

    column_index = {col: idx for idx, col in enumerate(train_df.columns)}
    smote = SMOTENC(
        sampling_strategy="auto",
        categorical_features=[column_index[c] for c in categorical if c in column_index],
    )
    resampled_X, resampled_y = smote.fit_resample(
        train_df.drop(LABEL, axis=1), train_df[LABEL])
    train_df = pd.concat([resampled_X, resampled_y], axis=1)
    for column in categorical:
        train_df[column] = train_df[column].astype(int)
        test_df[column] = test_df[column].astype(int)

    remove_illogical_values(train_df)
    remove_illogical_values(test_df)
    if len(train_df) == 0 or len(test_df) == 0:
        print(f"      [warn] fold {fold}: empty after remove_illogical; skipped.")
        return

    train_df, test_df, _, _, _ = scale_sample_train_and_test_df(
        train_df, test_df, None, categorical, continuous)
    features = [c for c in train_df.columns if c not in continuous] + continuous
    features.remove(LABEL)
    for column in categorical:
        train_df[column] = train_df[column].astype(int)
        test_df[column] = test_df[column].astype(int)

    country_arr = np.array([MALAYSIA] * n_msia_test + [INDIA] * n_india_test, dtype=int)
    fit_or_load_and_eval(
        strategy_name, fold,
        train_df[features], train_df[LABEL].astype(int),
        test_df[features], test_df[LABEL].astype(int),
        country_arr, all_results,
    )


def run_simple_imputation(msia_ds, india_ds, num_of_folds, all_results,
                          strategy="mean", strategy_name=None):
    """Strategies B and C: fill the absent cross-domain columns with a constant."""
    strategy_name = strategy_name or f"{strategy}_imputation"
    print(f"\n  Preparing {strategy}-imputation pipeline...")

    msia_df, msia_add_on, *_ = separate_df_and_df_add_on(msia_ds, LABEL, id_exists=True)
    india_df, india_add_on, *_ = separate_df_and_df_add_on(india_ds, LABEL, id_exists=False)
    selection = select_cross_domain_features(ACCURACY_THRESHOLD)
    categorical_candidates = selection.india_binary + selection.msia_multiclass + ["gender"]

    def simple_impute(target_df, source_train_df, features_to_add):
        out = target_df.copy()
        for feature in features_to_add:
            if feature in out.columns:
                continue
            if feature in CONSTANT_ZERO_FEATURES:
                out[feature] = 0
                continue
            if feature not in source_train_df.columns:
                out[feature] = 0
                continue
            source_values = source_train_df[feature].dropna()
            if len(source_values) == 0:
                out[feature] = 0
                continue
            modes = source_values.mode()
            if strategy == "mean":
                fill = (
                    (modes.iloc[0] if len(modes) else 0)
                    if feature in CATEGORICAL_FEATURES
                    else source_values.mean()
                )
            elif strategy == "mode":
                fill = modes.iloc[0] if len(modes) else source_values.mean()
            else:
                fill = 0
            out[feature] = fill
            if feature in CATEGORICAL_FEATURES:
                out[feature] = out[feature].round().astype(int)
        return out

    for fold in range(num_of_folds):
        print(f"\n    Fold {fold}:")
        msia_train, msia_test, india_train, india_test = _fold_frames(
            msia_df, msia_add_on, india_df, india_add_on, fold)

        india_train = simple_impute(india_train, msia_train, selection.msia_impute)
        india_test = simple_impute(india_test, msia_train, selection.msia_impute)
        msia_train = simple_impute(msia_train, india_train, selection.india_impute)
        msia_test = simple_impute(msia_test, india_train, selection.india_impute)

        for feature in selection.removed_from_malaysia:
            msia_train.drop(feature, axis=1, inplace=True, errors="ignore")
            msia_test.drop(feature, axis=1, inplace=True, errors="ignore")
        for feature in selection.removed_from_india:
            india_train.drop(feature, axis=1, inplace=True, errors="ignore")
            india_test.drop(feature, axis=1, inplace=True, errors="ignore")
        for frame in (msia_train, india_train, msia_test, india_test):
            frame.reset_index(drop=True, inplace=True)

        _finalize_and_eval(msia_train, msia_test, india_train, india_test,
                           categorical_candidates, strategy_name, fold, all_results)


def run_iterative_imputation(msia_ds, india_ds, num_of_folds, all_results,
                             strategy_name="E_iterative_imputation"):
    """Strategy E: MICE cross-imputation fitted on the pooled training rows."""
    print(f"\n  Preparing ITERATIVE-imputation pipeline ({strategy_name})...")
    msia_df, msia_add_on, *_ = separate_df_and_df_add_on(msia_ds, LABEL, id_exists=True)
    india_df, india_add_on, *_ = separate_df_and_df_add_on(india_ds, LABEL, id_exists=False)
    selection = select_cross_domain_features(ACCURACY_THRESHOLD)
    categorical_candidates = selection.india_binary + selection.msia_multiclass + ["gender"]
    removed = (selection.removed_from_malaysia + selection.removed_from_india)

    for fold in range(num_of_folds):
        print(f"\n    Fold {fold}:")
        msia_train, msia_test, india_train, india_test = _fold_frames(
            msia_df, msia_add_on, india_df, india_add_on, fold)

        for feature in selection.msia_impute:
            if feature not in india_train.columns:
                india_train[feature] = np.nan
                india_test[feature] = np.nan
        for feature in selection.india_impute:
            if feature not in msia_train.columns:
                value = 0 if feature in CONSTANT_ZERO_FEATURES else np.nan
                msia_train[feature] = value
                msia_test[feature] = value

        for feature in removed:
            for frame in (msia_train, msia_test, india_train, india_test):
                frame.drop(feature, axis=1, inplace=True, errors="ignore")
        for frame in (msia_train, india_train, msia_test, india_test):
            frame.reset_index(drop=True, inplace=True)

        columns = sorted(set(msia_train.columns) & set(india_train.columns))
        n_msia_test, n_india_test = len(msia_test), len(india_test)
        train_df = pd.concat([msia_train[columns], india_train[columns]],
                             axis=0).reset_index(drop=True)
        test_df = pd.concat([msia_test[columns], india_test[columns]],
                            axis=0).reset_index(drop=True)
        train_df[LABEL] = train_df[LABEL].astype(int)
        test_df[LABEL] = test_df[LABEL].astype(int)

        feature_cols = [c for c in columns if c != LABEL]
        categorical = [c for c in categorical_candidates if c in feature_cols]
        continuous = [c for c in feature_cols if c not in categorical]

        observed = train_df[feature_cols]
        minimums = observed.min(skipna=True).to_numpy(dtype=float)
        maximums = observed.max(skipna=True).to_numpy(dtype=float)
        for index in range(len(feature_cols)):
            if not np.isfinite(minimums[index]):
                minimums[index] = 0.0
            if not np.isfinite(maximums[index]):
                maximums[index] = 1.0
            if maximums[index] <= minimums[index]:
                maximums[index] = minimums[index] + 1e-6
        imputer = IterativeImputer(
            random_state=SEED, max_iter=10, sample_posterior=False,
            min_value=minimums, max_value=maximums,
        )
        train_df[feature_cols] = imputer.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = imputer.transform(test_df[feature_cols])
        for column in categorical:
            train_df[column] = train_df[column].round().clip(lower=0).astype(int)
            test_df[column] = test_df[column].round().clip(lower=0).astype(int)

        column_index = {col: idx for idx, col in enumerate(train_df.columns)}
        smote = SMOTENC(
            sampling_strategy="auto",
            categorical_features=[column_index[c] for c in categorical if c in column_index],
        )
        resampled_X, resampled_y = smote.fit_resample(
            train_df.drop(LABEL, axis=1), train_df[LABEL])
        train_df = pd.concat([resampled_X, resampled_y], axis=1)
        for column in categorical:
            train_df[column] = train_df[column].astype(int)
            test_df[column] = test_df[column].astype(int)

        remove_illogical_values(train_df)
        remove_illogical_values(test_df)
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"      [warn] fold {fold}: empty after remove_illogical "
                  f"(train={len(train_df)}, test={len(test_df)}); skipped.")
            continue

        train_df, test_df, _, _, _ = scale_sample_train_and_test_df(
            train_df, test_df, None, categorical, continuous)
        features = [c for c in train_df.columns if c not in continuous] + continuous
        features.remove(LABEL)
        for column in categorical:
            train_df[column] = train_df[column].astype(int)
            test_df[column] = test_df[column].astype(int)

        country_arr = np.array([MALAYSIA] * n_msia_test + [INDIA] * n_india_test, dtype=int)
        fit_or_load_and_eval(
            strategy_name, fold,
            train_df[features], train_df[LABEL].astype(int),
            test_df[features], test_df[LABEL].astype(int),
            country_arr, all_results,
        )


def run_experiment():
    """Run every imputation-strategy arm and write Table 7."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    msia_ds, india_ds = load_both_cohorts()
    print(f"Malaysia: {len(msia_ds[0])} samples (after fold 4 exclusion)")
    print(f"India: {len(india_ds[0])} samples (after fold 4 exclusion)")

    strategies = {
        "A_no_imputation": None,
        "B_mean_imputation": "mean",
        "C_mode_imputation": "mode",
        "D_default_catboost_imputation": "catboost",
        "E_iterative_imputation": "iterative",
    }
    all_results = []

    for strategy_name, strategy_param in strategies.items():
        print(f"\nStrategy: {strategy_name}")
        msia_arm, india_arm = load_both_cohorts()
        if strategy_param is None:
            run_no_imputation(msia_arm, india_arm, N_FOLDS_CV, all_results, strategy_name)
        elif strategy_param in ("mean", "mode"):
            run_simple_imputation(msia_arm, india_arm, N_FOLDS_CV, all_results,
                                  strategy=strategy_param, strategy_name=strategy_name)
        elif strategy_param == "catboost":
            run_catboost_imputation(msia_arm, india_arm, N_FOLDS_CV, all_results,
                                    strategy_name)
        elif strategy_param == "iterative":
            run_iterative_imputation(msia_arm, india_arm, N_FOLDS_CV, all_results,
                                     strategy_name)

    for selected_name, selected_features in SELECTED_IMPUTE_STRATEGIES.items():
        print(f"\nStrategy: {selected_name}")
        msia_arm, india_arm = load_both_cohorts()
        run_selected_catboost_imputation(msia_arm, india_arm, N_FOLDS_CV, all_results,
                                         selected_name, selected_features)

    results_df = pd.DataFrame(all_results).drop_duplicates().reset_index(drop=True)
    per_fold_path = SAVE_DIR / "per_fold_results.csv"
    results_df.to_csv(per_fold_path, index=False)
    print(f"\nPer-fold results (with eval_split) saved to: {per_fold_path}")
    print(f"Weights saved under: {WEIGHTS_DIR}/<strategy>/model_<fold>.pkl")

    summary_df = summarize(results_df, group_cols=("strategy", "eval_split"))
    summary_path = SAVE_DIR / "summary_comparison_by_country.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary (strategy x country) saved to: {summary_path}")

    for split in EVAL_SPLITS:
        subset = summary_df[summary_df["eval_split"] == split]
        if len(subset) == 0:
            continue
        print(f"\nIMPUTATION ABLATION -- split: {split.upper()} "
              f"(Logistic Regression, {N_FOLDS_CV}-fold CV)")
        print(f"{'Strategy':<22} {'Bal.Acc':<20} {'ROC AUC':<20} {'F1':<20} "
              f"{'AUPRC':<20} {'Brier':<20} {'ECE':<20}")
        print("-" * 142)
        for _, row in subset.iterrows():
            print(
                f"{row['strategy']:<22} "
                f"{row.get('balanced_accuracy_str', 'N/A'):<20} "
                f"{row.get('roc_auc_str', 'N/A'):<20} "
                f"{row.get('f1_str', 'N/A'):<20} "
                f"{row.get('auprc_str', 'N/A'):<20} "
                f"{row.get('brier_score_str', 'N/A'):<20} "
                f"{row.get('ece_str', 'N/A'):<20}"
            )

    print("\nExperiment complete.")
    return results_df, summary_df


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
