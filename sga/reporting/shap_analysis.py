"""SHAP attribution for the harmonized DNN (manuscript Figure 6)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHAP_DICT_KEYS = ("shap_df", "features", "shap_values", "test_data")


def new_shap_dict():
    """Empty accumulator with the keys `perform_SHAP` appends to."""
    return {key: [] for key in SHAP_DICT_KEYS}


def perform_SHAP(test_loader, model, features, shap_dict):
    """Compute DeepExplainer SHAP values for one fold and append them."""
    batch = next(iter(test_loader))
    data, _ = batch
    data = data.to(DEVICE)

    background = data[:100]
    test_data = data[100:]

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_data, check_additivity=False)
    shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])

    shap_df = pd.DataFrame(
        {
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
            "stdev_abs_shap": np.std(np.abs(shap_values), axis=0),
            "name": features,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    shap_dict["shap_df"].append(shap_df)
    shap_dict["features"].append(features)
    shap_dict["shap_values"].append(shap_values)
    shap_dict["test_data"].append(test_data)


def display_shap(shap_dict, download_path):
    """Write the per-fold SHAP tables and beeswarm summary plots."""
    os.makedirs(f"{download_path}/shap_results", exist_ok=True)
    os.makedirs(f"{download_path}/shap_plot", exist_ok=True)

    for i in range(len(shap_dict["shap_df"])):
        shap_save_path = f"{download_path}/shap_results/shap_values_{i}.xlsx"
        shap_plot_save_path = f"{download_path}/shap_plot/shap_summary_plot_{i}.png"

        shap_dict["shap_df"][i].to_excel(shap_save_path, index=False)

        feature_value = shap_dict["test_data"][i]
        shap.summary_plot(
            shap_dict["shap_values"][i],
            features=feature_value.cpu().numpy(),
            feature_names=shap_dict["features"][i],
            cmap="plasma",
            show=False,
        )
        plt.savefig(shap_plot_save_path, bbox_inches="tight")
        plt.close()


def load_shap_summaries(read_directory):
    """Read back the per-fold SHAP tables written by `display_shap`."""
    summaries = {}
    for file_name in sorted(os.listdir(read_directory)):
        shap_df = pd.read_excel(os.path.join(read_directory, file_name))
        summaries[file_name] = shap_df.sort_values("mean_abs_shap", ascending=False)
    return summaries
