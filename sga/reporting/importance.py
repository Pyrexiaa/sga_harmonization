"""Model-specific feature-importance tables and plots."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance

from sga.config import SEED


def compute_and_plot_permutation_importance(
    model, X_test, y_test, feature_names, fold, download_path, model_name
):
    """Compute and save permutation importance for any fitted estimator."""
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=SEED
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Permutation Importance": perm_importance.importances_mean,
        }
    ).sort_values(by="Permutation Importance", ascending=False)

    feature_dir = os.path.join(download_path, "feature_importances")
    plot_dir = os.path.join(download_path, "feature_importances_plotting")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    csv_file_path = os.path.join(
        feature_dir, f"permutation_importance_{model_name}_fold_{fold}.csv"
    )
    importance_df.to_csv(csv_file_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["Feature"],
        importance_df["Permutation Importance"],
        align="center",
        color="skyblue",
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Permutation Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"{model_name.upper()} Permutation Importance (Fold {fold})", fontsize=14)

    png_file_path = os.path.join(
        plot_dir, f"permutation_importance_{model_name}_fold_{fold}.png"
    )
    plt.savefig(png_file_path, bbox_inches="tight")
    plt.close()

    print(f"Permutation importance saved to:\n- CSV: {csv_file_path}\n- PNG: {png_file_path}")


def rf_f_importances(importances, feature_names, fold, download_path):
    """Save the random forest's Gini importances as a table and bar chart."""
    feature_imp_df = pd.DataFrame(
        {"Feature": feature_names, "Gini Importance": importances}
    ).sort_values("Gini Importance", ascending=False)
    print(feature_imp_df)
    os.makedirs(f"{download_path}/feature_importances", exist_ok=True)
    feature_imp_df.to_csv(f"{download_path}/feature_importances/rf_fold_{fold}.csv")

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_imp_df["Feature"], feature_imp_df["Gini Importance"], align="center"
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Gini Importance")
    plt.ylabel("Features")
    plt.title(f"Random Forest Feature Importance (Fold {fold})")

    os.makedirs(f"{download_path}/feature_importances_plotting", exist_ok=True)
    plt.savefig(
        f"{download_path}/feature_importances_plotting/rf_fold_{fold}.png",
        bbox_inches="tight",
    )
    print(f"Random Forest feature importance plot saved to 'rf_fold_{fold}.png'")
    plt.close()


def lr_f_importances(coefficients, feature_names, fold, download_path):
    """Save logistic-regression coefficients and odds ratios."""
    odds_ratios = np.exp(coefficients)
    print(len(feature_names))
    print(len(coefficients))
    print(len(odds_ratios))
    feature_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Odds Ratio": odds_ratios,
        }
    )
    print("\nFeature Importance (Coefficient and Odds Ratio):")
    sorted_feature_importance = feature_importance.sort_values(
        by="Coefficient", ascending=False
    )
    print(sorted_feature_importance)
    os.makedirs(f"{download_path}/feature_importances", exist_ok=True)
    sorted_feature_importance.to_csv(
        f"{download_path}/feature_importances/lr_fold_{fold}.csv"
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        sorted_feature_importance["Feature"],
        sorted_feature_importance["Coefficient"],
        align="center",
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient")
    plt.ylabel("Features")
    plt.title(f"Logistic Regression Coefficients (Fold {fold})")

    os.makedirs(f"{download_path}/feature_importances_plotting", exist_ok=True)
    plt.savefig(
        f"{download_path}/feature_importances_plotting/lr_fold_{fold}.png",
        bbox_inches="tight",
    )
    print(f"Logistic Regression feature importance plot saved to 'lr_fold_{fold}.png'")
    plt.close()


def svc_f_importances(perm_importance, features, fold, download_path):
    """Save pre-computed permutation importances for the support-vector model."""
    features = np.array(features)
    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Permutation Importance": perm_importance.importances_mean,
        }
    ).sort_values(by="Permutation Importance", ascending=False)

    os.makedirs(f"{download_path}/feature_importances", exist_ok=True)
    csv_file_path = f"{download_path}/feature_importances/svc_fold_{fold}.csv"
    importance_df.to_csv(csv_file_path, index=False)

    sorted_idx = perm_importance.importances_mean.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Permutation Importance of Features")

    os.makedirs(f"{download_path}/feature_importances_plotting", exist_ok=True)
    png_file_path = f"{download_path}/feature_importances_plotting/svc_fold_{fold}.png"
    plt.savefig(png_file_path, bbox_inches="tight")
    plt.close()


def catboost_f_importances(
    explainer, shap_values, train_X, features, fold, download_path, multiclass=False
):
    """Save CatBoost SHAP values, force plots and summary plots."""
    if multiclass:
        shap_values_flattened = shap_values.reshape(shap_values.shape[0], -1)

        num_classes = shap_values.shape[2]
        flattened_columns = [
            f"{feature}_class_{i}" for feature in features for i in range(num_classes)
        ]

        shap_values_df = pd.DataFrame(shap_values_flattened, columns=flattened_columns)
    else:
        shap_values_df = pd.DataFrame(shap_values, columns=features)

    os.makedirs(f"{download_path}/feature_importances", exist_ok=True)
    shap_values_csv_path = f"{download_path}/feature_importances/cb_fold_{fold}.csv"
    shap_values_df.to_csv(shap_values_csv_path, index=False)

    plot_dir = f"{download_path}/feature_importances_plotting"
    os.makedirs(plot_dir, exist_ok=True)

    def save_force_plot(index, filename, multiclass=False):
        if multiclass:
            for i in range(len(explainer.expected_value)):
                shap_html = shap.plots.force(
                    explainer.expected_value[i],
                    shap_values[index, :, i],
                    train_X.iloc[index, :],
                )
                class_filename = filename.replace(".html", f"_class_{i}.html")
                shap.save_html(class_filename, shap_html)
        else:
            shap_html = shap.plots.force(
                explainer.expected_value, shap_values[index, :], train_X.iloc[index, :]
            )
            shap.save_html(filename, shap_html)

    save_force_plot(
        0, f"{plot_dir}/cb_fold_{fold}_force_plot_0.html", multiclass=multiclass
    )
    save_force_plot(
        4, f"{plot_dir}/cb_fold_{fold}_force_plot_4.html", multiclass=multiclass
    )
    save_force_plot(
        slice(0, 50),
        f"{plot_dir}/cb_fold_{fold}_force_plot_0_50.html",
        multiclass=multiclass,
    )

    plt.figure(figsize=(10, 6))
    if multiclass:
        for i in range(num_classes):
            shap.summary_plot(shap_values[..., i], train_X, plot_type="bar", show=False)
            plt.savefig(
                f"{plot_dir}/cb_fold_{fold}_shap_summary_plot_class_{i}_bar.png",
                bbox_inches="tight",
            )
            plt.close()

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values[..., i], train_X, show=False)
            plt.savefig(
                f"{plot_dir}/cb_fold_{fold}_shap_summary_plot_class_{i}.png",
                bbox_inches="tight",
            )
            plt.close()
    else:
        shap.summary_plot(shap_values, train_X, plot_type="bar", show=False)
        plt.savefig(
            f"{plot_dir}/cb_fold_{fold}_shap_summary_plot_bar.png", bbox_inches="tight"
        )
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, train_X, show=False)
        plt.savefig(
            f"{plot_dir}/cb_fold_{fold}_shap_summary_plot.png", bbox_inches="tight"
        )
        plt.close()


def save_permutation_importance_list(permutation_importance_list, download_path):
    """Write a collected set of permutation-importance frames and bar charts."""
    for idx, permutation_importance_df in enumerate(permutation_importance_list):
        plot_save_path = f"{download_path}/permutation_importance/plot_{idx}.png"
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

        csv_save_path = f"{download_path}/permutation_importance/data_{idx}.csv"
        permutation_importance_df.to_csv(csv_save_path, index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=permutation_importance_df,
            x="importance",
            y="feature",
            palette="viridis",
        )
        plt.xlabel("Permutation Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.title(f"Permutation Importance of Features (Plot {idx})", fontsize=14)

        plt.savefig(plot_save_path, bbox_inches="tight")
        plt.close()
