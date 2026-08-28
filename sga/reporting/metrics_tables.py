"""Per-fold metric computation and the metric spreadsheets they are written to."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)

from sga.config import N_BOOTSTRAP, SEED


def calculate_metrics(cm):
    """Macro-averaged PPV, NPV, sensitivity and specificity from a confusion matrix."""
    cm = np.array(cm)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}")

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    PPV = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
    NPV = np.divide(TN, TN + FN, out=np.zeros_like(TN, dtype=float), where=(TN + FN) != 0)
    Sensitivity = np.divide(
        TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0
    )
    Specificity = np.divide(
        TN, TN + FP, out=np.zeros_like(TN, dtype=float), where=(TN + FP) != 0
    )

    return np.mean(PPV), np.mean(NPV), np.mean(Sensitivity), np.mean(Specificity)


def calc_metrics_updated(y_true, y_pred, y_pred_prob=None, metrics=(), multiclass=False):
    """Evaluate metric callables against one fold's predictions."""
    out = []
    for met in metrics:
        if (y_pred_prob is not None and met == roc_auc_score) or met == roc_curve:
            if multiclass and met == roc_auc_score:
                out.append(met(y_true, y_pred_prob, multi_class="ovr"))
            elif multiclass and met == roc_curve:
                continue
            else:
                out.append(met(y_true, y_pred_prob))
        elif multiclass and met != balanced_accuracy_score:
            out.append(met(y_true, y_pred, average="weighted"))
        else:
            out.append(met(y_true, y_pred))
    return out


def calc_metrics_with_ci(
    y_true,
    y_pred,
    y_pred_prob=None,
    metrics=(),
    multiclass=False,
    n_bootstrap=N_BOOTSTRAP,
    ci=0.95,
    download_path=None,
    fold=0,
    training=False,
    save=True,
    seed=SEED,
):
    """Evaluate metrics and bootstrap a confidence interval for each scalar one."""
    results = {}
    out = []
    rng = np.random.RandomState(seed)

    for met in metrics:
        metric_name = met.__name__
        if y_pred_prob is not None and met in [roc_auc_score, roc_curve]:
            if multiclass and met == roc_auc_score:
                score = met(y_true, y_pred_prob, multi_class="ovr")
                out.append(score)
            elif met == roc_auc_score:
                try:
                    score = met(y_true, y_pred_prob)
                    out.append(score)
                except ValueError:
                    out.append(float("nan"))
            elif met == roc_curve:
                if not multiclass:
                    try:
                        fpr, tpr, _ = met(y_true, y_pred_prob)
                        score = (fpr, tpr)
                        out.append(score)
                    except ValueError:
                        out.append(float("nan"))
                else:
                    continue
            else:
                score = met(y_true, y_pred_prob)
                out.append(score)
        elif multiclass and met != balanced_accuracy_score:
            score = met(y_true, y_pred, average="weighted")
            out.append(score)
        elif met == precision_score:
            score = met(y_true, y_pred, zero_division=0)
            out.append(score)
        else:
            score = met(y_true, y_pred)
            out.append(score)

        if metric_name == "roc_curve":
            continue

        bootstrap_scores = []
        for _ in range(n_bootstrap):
            indices = rng.choice(len(y_true), size=len(y_true), replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_pred_prob_boot = y_pred_prob[indices] if y_pred_prob is not None else None

            if y_pred_prob_boot is not None and met in [roc_auc_score, roc_curve]:
                if multiclass and met == roc_auc_score:
                    boot_score = met(y_true_boot, y_pred_prob_boot, multi_class="ovr")
                elif met == roc_auc_score:
                    try:
                        boot_score = met(y_true_boot, y_pred_prob_boot)
                    except ValueError:
                        boot_score = float("nan")
                elif met == roc_curve and not multiclass:
                    fpr, tpr, _ = met(y_true_boot, y_pred_prob_boot)
                    boot_score = (fpr, tpr)
                else:
                    continue
            elif multiclass and met != balanced_accuracy_score:
                boot_score = met(y_true_boot, y_pred_boot, average="weighted")
            else:
                boot_score = met(y_true_boot, y_pred_boot)

            bootstrap_scores.append(boot_score)

        if isinstance(score, (int, float)):
            lower_bound = np.percentile(bootstrap_scores, (1 - ci) / 2 * 100)
            upper_bound = np.percentile(bootstrap_scores, (1 + ci) / 2 * 100)
            results[metric_name] = {
                "value": score,
                f"{int(ci * 100)}% CI": (lower_bound, upper_bound),
            }
        else:
            results[metric_name] = {"value": score}

    ci_df = pd.DataFrame(results)
    if training:
        path = f"{download_path}/metrics/train_95_confidence_interval_result_{fold}"
    else:
        path = f"{download_path}/metrics/test_95_confidence_interval_result_{fold}"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if save:
        ci_df.to_csv(f"{path}.csv", index=False)
        ci_df.to_excel(f"{path}.xlsx", index=False, engine="openpyxl")

    return out


def calc_multiclassification_metrics(y_true, y_pred, y_pred_prob, metrics=()):
    """Micro-averaged metrics for the multiclass SGA head."""
    out = []
    for met in metrics:
        if met == roc_auc_score:
            out.append(met(y_true, y_pred_prob, multi_class="ovr", average="micro"))
        elif met == balanced_accuracy_score:
            out.append(met(y_true, y_pred))
        else:
            out.append(met(y_true, y_pred, average="micro"))
    return out


def display_metrics(
    models,
    num_of_folds,
    download_path,
    training_balanced_accuracy,
    validation_balanced_accuracy,
    validation_oof_acc,
    validation_oof_roc_auc,
    validation_oof_f1,
    validation_oof_prec,
    validation_oof_rec,
):
    """Write the per-fold validation metric table for a DNN run."""
    data = {
        "Fold": [],
        "Training Balanced Accuracy": [],
        "Balanced Accuracy": [],
        "OOF Acc": [],
        "OOF ROC AUC": [],
        "OOF F1": [],
        "OOF Precision": [],
        "OOF Recall": [],
    }

    for i in range(num_of_folds):
        data["Fold"].append(i)
        if len(training_balanced_accuracy) > 0:
            data["Training Balanced Accuracy"].append(
                f"{training_balanced_accuracy[i]:.4f}"
            )
        data["Balanced Accuracy"].append(f"{validation_balanced_accuracy[i]:.4f}")
        data["OOF Acc"].append(f"{validation_oof_acc[i]:.4f}")
        data["OOF ROC AUC"].append(f"{validation_oof_roc_auc[i]:.4f}")
        data["OOF F1"].append(f"{validation_oof_f1[i]:.4f}")
        data["OOF Precision"].append(f"{validation_oof_prec[i]:.4f}")
        data["OOF Recall"].append(f"{validation_oof_rec[i]:.4f}")

        save_path = f"{download_path}/model_weights/model_{i}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(models[i].state_dict(), save_path)

    data["Fold"].append("Mean")
    if len(training_balanced_accuracy) > 0:
        data["Training Balanced Accuracy"].append(
            f"{np.mean(training_balanced_accuracy):.4f}"
        )
    data["Balanced Accuracy"].append(f"{np.mean(validation_balanced_accuracy):.4f}")
    data["OOF Acc"].append(f"{np.mean(validation_oof_acc):.4f}")
    data["OOF ROC AUC"].append(f"{np.mean(validation_oof_roc_auc):.4f}")
    data["OOF F1"].append(f"{np.mean(validation_oof_f1):.4f}")
    data["OOF Precision"].append(f"{np.mean(validation_oof_prec):.4f}")
    data["OOF Recall"].append(f"{np.mean(validation_oof_rec):.4f}")

    if len(training_balanced_accuracy) == 0:
        del data["Training Balanced Accuracy"]
    save_path = f"{download_path}/metrics/model_metrics.xlsx"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(data).to_excel(save_path, index=False, engine="openpyxl")


def display_metrics_updated(
    models,
    num_of_folds,
    download_path,
    training_balanced_accuracy,
    training_oof_acc,
    training_oof_roc_auc,
    training_oof_f1,
    training_oof_prec,
    training_oof_rec,
    validation_balanced_accuracy,
    validation_oof_acc,
    validation_oof_roc_auc,
    validation_oof_f1,
    validation_oof_prec,
    validation_oof_rec,
):
    """Write the per-fold training and validation metric table for a DNN run."""
    data = {
        "Fold": [],
        "Training Balanced Accuracy": [],
        "Training OOF Acc": [],
        "Training OOF ROC AUC": [],
        "Training OOF F1": [],
        "Training OOF Precision": [],
        "Training OOF Recall": [],
        "Balanced Accuracy": [],
        "OOF Acc": [],
        "OOF ROC AUC": [],
        "OOF F1": [],
        "OOF Precision": [],
        "OOF Recall": [],
    }

    for i in range(num_of_folds):
        data["Fold"].append(i)
        if len(training_balanced_accuracy) > 0:
            data["Training Balanced Accuracy"].append(
                f"{training_balanced_accuracy[i]:.4f}"
            )
            data["Training OOF Acc"].append(f"{training_oof_acc[i]:.4f}")
            data["Training OOF ROC AUC"].append(f"{training_oof_roc_auc[i]:.4f}")
            data["Training OOF F1"].append(f"{training_oof_f1[i]:.4f}")
            data["Training OOF Precision"].append(f"{training_oof_prec[i]:.4f}")
            data["Training OOF Recall"].append(f"{training_oof_rec[i]:.4f}")

        data["Balanced Accuracy"].append(f"{validation_balanced_accuracy[i]:.4f}")
        data["OOF Acc"].append(f"{validation_oof_acc[i]:.4f}")
        data["OOF ROC AUC"].append(f"{validation_oof_roc_auc[i]:.4f}")
        data["OOF F1"].append(f"{validation_oof_f1[i]:.4f}")
        data["OOF Precision"].append(f"{validation_oof_prec[i]:.4f}")
        data["OOF Recall"].append(f"{validation_oof_rec[i]:.4f}")

        save_path = f"{download_path}/model_weights/model_{i}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(models[i].state_dict(), save_path)

    data["Fold"].append("Mean")
    if len(training_balanced_accuracy) > 0:
        data["Training Balanced Accuracy"].append(
            f"{np.mean(training_balanced_accuracy):.4f}"
        )
        data["Training OOF Acc"].append(f"{np.mean(training_oof_acc):.4f}")
        data["Training OOF ROC AUC"].append(f"{np.mean(training_oof_roc_auc):.4f}")
        data["Training OOF F1"].append(f"{np.mean(training_oof_f1):.4f}")
        data["Training OOF Precision"].append(f"{np.mean(training_oof_prec):.4f}")
        data["Training OOF Recall"].append(f"{np.mean(training_oof_rec):.4f}")

    data["Balanced Accuracy"].append(f"{np.mean(validation_balanced_accuracy):.4f}")
    data["OOF Acc"].append(f"{np.mean(validation_oof_acc):.4f}")
    data["OOF ROC AUC"].append(f"{np.mean(validation_oof_roc_auc):.4f}")
    data["OOF F1"].append(f"{np.mean(validation_oof_f1):.4f}")
    data["OOF Precision"].append(f"{np.mean(validation_oof_prec):.4f}")
    data["OOF Recall"].append(f"{np.mean(validation_oof_rec):.4f}")

    if len(training_balanced_accuracy) == 0:
        del data["Training Balanced Accuracy"]
    save_path = f"{download_path}/metrics/model_metrics.xlsx"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(data).to_excel(save_path, index=False, engine="openpyxl")


def display_ml_metrics(
    models,
    num_of_folds,
    download_path,
    validation_balanced_accuracy,
    validation_oof_roc_auc,
    validation_oof_f1,
    validation_oof_prec,
    validation_oof_rec,
    catboost=False,
):
    """Write the per-fold metric table for a classical-ML or CatBoost run."""
    data = {
        "Fold": [],
        "OOF Balanced Accuracy": [],
        "OOF ROC AUC": [],
        "OOF F1": [],
        "OOF Precision": [],
        "OOF Recall": [],
    }

    for i in range(num_of_folds):
        data["Fold"].append(i)
        data["OOF Balanced Accuracy"].append(f"{validation_balanced_accuracy[i]:.4f}")
        if len(validation_oof_roc_auc) > 0:
            data["OOF ROC AUC"].append(f"{validation_oof_roc_auc[i]:.4f}")
        else:
            data["OOF ROC AUC"].append(0)
        data["OOF F1"].append(f"{validation_oof_f1[i]:.4f}")
        data["OOF Precision"].append(f"{validation_oof_prec[i]:.4f}")
        data["OOF Recall"].append(f"{validation_oof_rec[i]:.4f}")

        save_path = f"{download_path}/model_weights/model_{i}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if catboost:
            models[i].save_model(save_path[:-4])
        else:
            with open(save_path, "wb") as handle:
                pickle.dump(models[i], handle)

    data["Fold"].append("Mean")
    data["OOF Balanced Accuracy"].append(f"{np.mean(validation_balanced_accuracy):.4f}")
    data["OOF ROC AUC"].append(f"{np.mean(validation_oof_roc_auc):.4f}")
    data["OOF F1"].append(f"{np.mean(validation_oof_f1):.4f}")
    data["OOF Precision"].append(f"{np.mean(validation_oof_prec):.4f}")
    data["OOF Recall"].append(f"{np.mean(validation_oof_rec):.4f}")

    if len(validation_oof_roc_auc) == 0:
        del data["OOF ROC AUC"]

    save_path = f"{download_path}/metrics/model_metrics.xlsx"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(data).to_excel(save_path, index=False, engine="openpyxl")


def display_ml_regression_metrics(
    models,
    num_of_folds,
    download_path,
    validation_mae,
    validation_rmse,
    validation_r2,
    catboost=False,
):
    """Write the per-fold error table for the continuous cross-domain imputers."""
    data = {
        "Fold": [],
        "OOF MAE": [],
        "OOF RMSE": [],
        "OOF R2 Score": [],
    }

    for i in range(num_of_folds):
        data["Fold"].append(i)
        data["OOF MAE"].append(f"{validation_mae[i]:.4f}")
        data["OOF RMSE"].append(f"{validation_rmse[i]:.4f}")
        data["OOF R2 Score"].append(f"{validation_r2[i]:.4f}")

        save_path = f"{download_path}/model_weights/model_{i}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if catboost:
            models[i].save_model(save_path[:-4])
        else:
            with open(save_path, "wb") as handle:
                pickle.dump(models[i], handle)

    data["Fold"].append("Mean")
    data["OOF MAE"].append(f"{np.mean(validation_mae):.4f}")
    data["OOF RMSE"].append(f"{np.mean(validation_rmse):.4f}")
    data["OOF R2 Score"].append(f"{np.mean(validation_r2):.4f}")

    save_path = f"{download_path}/metrics/model_metrics.xlsx"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(data).to_excel(save_path, index=False, engine="openpyxl")


def display_metrics_for_multiple_iterations(
    models_list,
    num_of_folds,
    download_paths,
    validation_balanced_accuracy_list,
    validation_oof_acc_list,
    validation_oof_roc_auc_list,
    validation_oof_f1_list,
    validation_oof_prec_list,
    validation_oof_rec_list,
):
    """Average per-fold metrics across repeated runs of the same configuration."""
    mean_metrics = {
        "Fold": [],
        "Mean Balanced Accuracy": [],
        "Mean OOF Acc": [],
        "Mean OOF ROC AUC": [],
        "Mean OOF F1": [],
        "Mean OOF Precision": [],
        "Mean OOF Recall": [],
    }

    for i, (
        models,
        download_path,
        balanced_accuracy,
        acc,
        roc_auc,
        f1,
        prec,
        rec,
    ) in enumerate(
        zip(
            models_list,
            download_paths,
            validation_balanced_accuracy_list,
            validation_oof_acc_list,
            validation_oof_roc_auc_list,
            validation_oof_f1_list,
            validation_oof_prec_list,
            validation_oof_rec_list,
        )
    ):
        mean_balanced_accuracy = np.mean(
            [item for sublist in balanced_accuracy for item in sublist]
        )
        mean_acc = np.mean([item for sublist in acc for item in sublist])
        mean_roc_auc = np.mean([item for sublist in roc_auc for item in sublist])
        mean_f1 = np.mean([item for sublist in f1 for item in sublist])
        mean_prec = np.mean([item for sublist in prec for item in sublist])
        mean_rec = np.mean([item for sublist in rec for item in sublist])

        mean_metrics["Fold"].append(f"Loop {i}")
        mean_metrics["Mean Balanced Accuracy"].append(mean_balanced_accuracy)
        mean_metrics["Mean OOF Acc"].append(mean_acc)
        mean_metrics["Mean OOF ROC AUC"].append(mean_roc_auc)
        mean_metrics["Mean OOF F1"].append(mean_f1)
        mean_metrics["Mean OOF Precision"].append(mean_prec)
        mean_metrics["Mean OOF Recall"].append(mean_rec)

        for k in range(num_of_folds):
            save_path = f"{download_path}_loop_{i}/model_weights/model_{k}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(models[k].state_dict(), save_path)

    mean_save_path = f"{download_paths[0]}_loop_mean/model_metrics_mean.xlsx"
    os.makedirs(os.path.dirname(mean_save_path), exist_ok=True)
    pd.DataFrame(mean_metrics).to_excel(
        mean_save_path, index=False, engine="openpyxl"
    )
