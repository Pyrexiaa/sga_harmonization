"""SHAP feature attribution per cohort, without retraining (manuscript Figure 6).

    Run experiment_R0_baseline_retrain.py, then
    experiment_R0_baseline_retrain_manual.py, and
    experiment_R1_1_remove_prev_pregnancy.py.

Run:
    python -m rebuttals.round1.shap_without_retraining
"""

from __future__ import annotations

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from sga.config import (
    COMMON_FEATURES,
    HARMONIZED_SELECTED_FEATURES,
    N_FOLDS_CV,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.models.torch_utils import DEVICE
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA
from sga.pipeline.model_io import load_trained_model

from rebuttals.round1.evaluate_per_country import build_variant_folds
from rebuttals.round1.experiment_R0_baseline_retrain import (
    HARMONIZED_WEIGHTS_DIR,
    SAVE_DIR as R0_DIR,
)

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND1_DIR / "per_country_shap_no_retrain"

# Caps that keep KernelExplainer tractable.
MAX_EXPLAIN = 300      # rows explained per cohort per fold
MAX_BACKGROUND = 100   # background reference sample size

# The harmonized feature space reported in Figure 6.
WITHOUT_PP_FEATURES = list(COMMON_FEATURES) + list(HARMONIZED_SELECTED_FEATURES)

SHAP_TARGETS = [
    dict(
        name="Logistic Regression (Without Poorly Imputed Features)",
        family="ml",
        variant="without_prev_preg",
        path=str(HARMONIZED_WEIGHTS_DIR / f"generalized_lr_{SEED}" / "malaysia_tri3"),
    ),
    dict(
        name="Logistic Regression (With Previous Pregnancy Features)",
        family="ml",
        variant="with_prev_preg",
        path=str(R0_DIR / f"generalized_lr_{SEED}" / "malaysia_tri3"),
    ),
]


def _subsample(frame, n, seed):
    """Random row subsample, or the frame unchanged when it is already small."""
    return frame if len(frame) <= n else frame.sample(n=n, random_state=seed)


def shap_values_for_subset(model, family, background_X, explain_X, features):
    """SHAP values for ``explain_X`` against a background reference set."""
    if len(explain_X) == 0:
        return np.empty((0, len(features)))

    if family == "dnn":
        background = torch.as_tensor(
            background_X.values, dtype=torch.float32, device=DEVICE)
        explain = torch.as_tensor(explain_X.values, dtype=torch.float32, device=DEVICE)
        values = shap.DeepExplainer(model, background).shap_values(
            explain, check_additivity=False)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values.reshape(values.shape[0], values.shape[1])
        return values

    if family == "catboost":
        values = shap.TreeExplainer(model).shap_values(explain_X)
        if isinstance(values, list):  # some versions return [class0, class1]
            values = values[-1]
        values = np.asarray(values)
        if values.shape[1] == len(features) + 1:  # drop the bias column
            values = values[:, :-1]
        return values

    explainer = shap.KernelExplainer(
        lambda x: model.predict_proba(x)[:, 1], background_X)
    values = np.asarray(
        explainer.shap_values(explain_X, nsamples="auto", silent=True))
    if values.ndim == 3:
        values = values[..., -1]
    return values


def mean_abs(shap_values, features):
    """Mean and standard deviation of |SHAP| per feature."""
    if len(shap_values) == 0:
        return pd.DataFrame({
            "name": features,
            "mean_abs_shap": [np.nan] * len(features),
            "stdev_abs_shap": [np.nan] * len(features),
        })
    return pd.DataFrame({
        "name": features,
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        "stdev_abs_shap": np.std(np.abs(shap_values), axis=0),
    })


def bar_plot(aggregated, title, out_png):
    """Horizontal bar chart of the top-20 features by mean |SHAP|."""
    top = aggregated.sort_values("mean_abs_shap", ascending=True).tail(20)
    plt.figure(figsize=(8, max(3, 0.35 * len(top))))
    plt.barh(top["name"], top["mean_abs_shap"])
    plt.xlabel("SHAP value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def run():
    """Compute and save the per-cohort SHAP rankings for every target model."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for target in SHAP_TARGETS:
        print(f"\nSHAP :: {target['name']}  [{target['family']}]  ({target['variant']})")
        restrict_to_harmonized = target["variant"] == "without_prev_preg"
        folds = build_variant_folds(target["variant"])

        per_fold_rows = []
        cohort_mean_abs = {"malaysia": [], "india": []}

        for fold in range(N_FOLDS_CV):
            test_X, _, country_arr, features = folds[fold]

            # Trim BEFORE loading so the DNN input width and the SHAP inputs match the
            # columns the model was trained on.
            if restrict_to_harmonized:
                keep = [f for f in features if f in set(WITHOUT_PP_FEATURES)]
                missing = [f for f in WITHOUT_PP_FEATURES if f not in features]
                if missing:
                    print(f"  [fold {fold}] warning: expected harmonized features "
                          f"missing from the reconstruction: {missing}")
                test_X = test_X[keep]
                features = keep
                print(f"  [fold {fold}] without_prev_preg -> {len(features)} "
                      f"features: {features}")

            model = load_trained_model(
                target["family"], target["path"], fold,
                n_features=len(features), dnn_config=target.get("dnn_config"),
            )
            if model is None:
                print(f"  [fold {fold}] no saved weights -- skipped "
                      f"({target['path']}/model_weights/model_{fold}*)")
                continue

            background_X = _subsample(test_X, MAX_BACKGROUND, seed=SEED + fold)
            for cohort, code in (("malaysia", MALAYSIA), ("india", INDIA)):
                subset = _subsample(
                    test_X[np.asarray(country_arr) == code], MAX_EXPLAIN, seed=SEED + fold)
                if len(subset) == 0:
                    print(f"  [fold {fold}] {cohort}: 0 rows -- skipped")
                    continue
                try:
                    values = shap_values_for_subset(
                        model, target["family"], background_X, subset, features)
                except Exception as error:  # noqa: BLE001
                    print(f"  [fold {fold}] {cohort}: SHAP failed: {error}")
                    continue

                frame = mean_abs(values, features)
                frame.insert(0, "fold", fold)
                frame.insert(0, "country", cohort)
                per_fold_rows.append(frame)
                cohort_mean_abs[cohort].append(frame.set_index("name")["mean_abs_shap"])
                top_feature = frame.sort_values(
                    "mean_abs_shap", ascending=False).iloc[0]["name"]
                print(f"  [fold {fold}] {cohort}: SHAP on {len(subset)} rows, "
                      f"top feat = {top_feature}")

        if per_fold_rows:
            long_path = SAVE_DIR / f"shap_per_fold_{target['name']}.csv"
            pd.concat(per_fold_rows, ignore_index=True).to_csv(long_path, index=False)
            print(f"  saved {long_path}")

        for cohort in ("malaysia", "india"):
            series = cohort_mean_abs[cohort]
            if not series:
                continue
            joined = pd.concat(series, axis=1)
            aggregated = (
                pd.DataFrame({
                    "name": joined.index,
                    "mean_abs_shap": joined.mean(axis=1).values,
                    "std_across_folds": joined.std(axis=1).values,
                })
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            out_csv = SAVE_DIR / f"shap_meanabs_{target['name']}_{cohort}.csv"
            aggregated.to_csv(out_csv, index=False)
            bar_plot(
                aggregated,
                f"{target['name']} - {cohort.capitalize()}",
                os.path.join(str(SAVE_DIR), f"shap_bar_{target['name']}_{cohort}.png"),
            )
            print(f"  saved {out_csv}")

    print(f"\nSHAP-by-country complete. Results in: {SAVE_DIR}")


if __name__ == "__main__":
    set_seed(SEED)
    run()
