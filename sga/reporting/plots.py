"""Per-fold diagnostic plots written alongside the metric tables."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve


def plot_confusion_matrix(cm, classes, download_path):
    """Save a single annotated confusion-matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(download_path)
    plt.close()


def display_roc_curve(roc_curve_result, download_path, training=False):
    """Plot pre-computed ROC curves for every fold and tabulate their points."""
    frames = []
    for i, curve in enumerate(roc_curve_result):
        if training:
            save_path = f"{download_path}/roc_curve/training_roc_curve_plot_{i}.png"
        else:
            save_path = f"{download_path}/roc_curve/roc_curve_plot_{i}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        lr_fpr, lr_tpr, _ = curve
        plt.plot(lr_fpr, lr_tpr, marker=".", label=f"Model ROC Curve {i}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(save_path)

        frames.append(
            pd.DataFrame(
                {
                    "Model": [f"Model {i}"] * len(lr_fpr),
                    "False Positive Rate": lr_fpr,
                    "True Positive Rate": lr_tpr,
                }
            )
        )

    if training:
        excel_save_path = f"{download_path}/roc_curve/train_roc_curve_data.xlsx"
    else:
        excel_save_path = f"{download_path}/roc_curve/roc_curve_data.xlsx"
    pd.concat(frames, ignore_index=True).to_excel(excel_save_path, index=False)


def display_roc_curve_binary(
    roc_curve_result, y_true_list, y_score_list, download_path, training=False
):
    """Plot overall and per-class ROC curves for a binary task, fold by fold."""
    roc_data = pd.DataFrame(
        columns=["Model", "Class", "False Positive Rate", "True Positive Rate"]
    )

    for i, (y_true, y_score) in enumerate(zip(y_true_list, y_score_list)):
        if training:
            save_path = f"{download_path}/roc_curve/training_roc_curve_plot_{i}.png"
            binary_class_save_path = f"{download_path}/roc_curve/training_binary_class_roc_curve_plot_{i}.png"
        else:
            save_path = f"{download_path}/roc_curve/roc_curve_plot_{i}.png"
            binary_class_save_path = (
                f"{download_path}/roc_curve/binary_class_roc_curve_plot_{i}.png"
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        auc_score = auc(fpr, tpr)
        plt.plot(
            fpr, tpr, marker=".", label=f"Model ROC Curve {i} (AUC = {auc_score:.2f})"
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(save_path)
        plt.clf()

        temp_df = pd.DataFrame(
            {
                "Model": [f"Model {i}"] * len(fpr),
                "Class": ["Overall"] * len(fpr),
                "False Positive Rate": fpr,
                "True Positive Rate": tpr,
            }
        )
        roc_data = pd.concat([roc_data, temp_df], ignore_index=True)

        fpr_1, tpr_1, _ = roc_curve(y_true, y_score, pos_label=1)
        auc_1 = auc(fpr_1, tpr_1)

        fpr_0, tpr_0, _ = roc_curve(y_true, 1 - y_score, pos_label=0)
        auc_0 = auc(fpr_0, tpr_0)

        class_1_df = pd.DataFrame(
            {
                "Model": [f"Model {i}"] * len(fpr_1),
                "Class": ["Class 1"] * len(fpr_1),
                "False Positive Rate": fpr_1,
                "True Positive Rate": tpr_1,
            }
        )
        class_0_df = pd.DataFrame(
            {
                "Model": [f"Model {i}"] * len(fpr_0),
                "Class": ["Class 0"] * len(fpr_0),
                "False Positive Rate": fpr_0,
                "True Positive Rate": tpr_0,
            }
        )
        roc_data = pd.concat([roc_data, class_1_df, class_0_df], ignore_index=True)

        plt.plot(fpr_1, tpr_1, color="blue", lw=2, label=f"Class 1 (AUC = {auc_1:.2f})")
        plt.plot(fpr_0, tpr_0, color="red", lw=2, label=f"Class 0 (AUC = {auc_0:.2f})")
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Binary Class ROC Curve for Model {i}")
        plt.legend(loc="lower right")
        plt.savefig(binary_class_save_path)
        plt.clf()

    if training:
        excel_save_path = f"{download_path}/roc_curve/train_roc_curve_data.xlsx"
    else:
        excel_save_path = f"{download_path}/roc_curve/roc_curve_data.xlsx"
    roc_data.to_excel(excel_save_path, index=False)


def display_cm(overall_cm_list, download_path, training=False):
    """Save one confusion-matrix heatmap per fold."""
    for i, overall_cm in enumerate(overall_cm_list):
        if training:
            save_path = f"{download_path}/confusion_matrix/training_confusion_matrix_plot_{i}.png"
        else:
            save_path = (
                f"{download_path}/confusion_matrix/confusion_matrix_plot_{i}.png"
            )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_confusion_matrix(overall_cm, ["Class 0", "Class 1"], save_path)


def display_training_loss(epoch_nums, training_loss, download_path):
    """Save the training-loss trace of each fold."""
    for idx, loss_list in enumerate(training_loss):
        plt.plot(
            epoch_nums[idx], loss_list, label=f"Training Loss Fold {idx}", color="red"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Training Loss vs. Epochs (Fold {idx})")
        save_path = f"{download_path}/training_loss/training_loss_graph_{idx}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()


def display_validation_loss(epoch_nums, validation_loss, download_path):
    """Save the validation-loss trace of each fold."""
    for idx, loss_list in enumerate(validation_loss):
        plt.plot(
            epoch_nums[idx], loss_list, label=f"Validation Loss Fold {idx}", color="red"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Validation Loss vs. Epochs (Fold {idx})")
        save_path = f"{download_path}/validation_loss/validation_loss_graph_{idx + 1}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()


def display_trees(tree_dict, download_path):
    """Report the identifier of the model whose trees were exported."""
    print("Model ID: ", tree_dict["model_id"])
