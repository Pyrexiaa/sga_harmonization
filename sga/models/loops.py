"""Per-epoch train, validate and test loops for the binary DNN classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from sga.reporting.metrics_tables import (
    calc_metrics_updated,
    calc_metrics_with_ci,
    calculate_metrics,
)
from sga.models.torch_utils import CLASSIFICATION_THRESHOLD, DEVICE

METRICS_LIST = [balanced_accuracy_score, roc_auc_score, f1_score,
                precision_score, recall_score, roc_curve]

# Intermediate representations available for each model size.
LAYER_OUTPUTS = {
    'large': ['out_after_first_layer', 'out_after_second_layer', 'out_after_third_layer'],
    'medium': ['out_after_first_layer', 'out_after_second_layer'],
    'small': ['out_after_first_layer'],
    'test': ['out_after_first_layer', 'out_after_second_layer', 'out_after_third_layer'],
    'calibration': ['out_after_first_layer', 'out_after_second_layer'],
}


def L1_loss(model):
    """Compute the normalised L1 regularisation term over non-bias parameters."""
    nweights = sum(w.numel() for name, w in model.named_parameters() if 'bias' not in name)
    L1_term = torch.tensor(0., requires_grad=True)
    for name, weights in model.named_parameters():
        if 'bias' not in name:
            L1_term = L1_term + torch.sum(torch.abs(weights))
    return L1_term / nweights


def _collect_predictions(actual_tensors, pred_tensors, prob_tensors, data_tensors):
    """Convert the per-batch tensor lists to NumPy arrays for metric computation."""
    actuals = np.array([t.item() for t in actual_tensors])
    predictions = np.array([t.item() for t in pred_tensors])
    all_data = torch.cat(data_tensors, dim=0).numpy()
    probabilities = np.array([p.detach().numpy() for p in prob_tensors])
    return actuals, predictions, all_data, probabilities


def _convert_tensors_to_dataframe(output_dict):
    """Convert a dict of tensor lists to a dict of DataFrames."""
    return {
        key: pd.DataFrame(torch.cat(tensors, dim=0).cpu().detach().numpy())
        for key, tensors in output_dict.items()
    }


def _extract_layer_outputs(model, data, output_dict):
    """Append the requested intermediate layer outputs for one batch."""
    layer_extractors = {
        'out_after_first_layer': model.get_x_after_first_layer,
        'out_after_second_layer': model.get_x_after_second_layer,
        'out_after_third_layer': model.get_x_after_third_layer,
    }
    for key in output_dict:
        output_dict[key].append(layer_extractors[key](data))


def _print_final_metrics(acc, roc_auc, f1, prec, rec, overall_cm):
    """Print the confusion-matrix-derived and out-of-fold metrics."""
    PPV, NPV, Sensitivity, Specificity = calculate_metrics(overall_cm)
    print(f"PPV: {PPV:.4f}, NPV: {NPV:.4f}, "
          f"Sens: {Sensitivity:.4f}, Spec: {Specificity:.4f}")
    print(f"OOF Balanced Acc: {np.mean(acc):.4f}, "
          f"ROC AUC: {np.mean(roc_auc):.4f}, F1: {np.mean(f1):.4f}, "
          f"Prec: {np.mean(prec):.4f}, Recall: {np.mean(rec):.4f}")


def train_dnn(model, data_loader, optimizer, loss_criteria, features=None,
              l1_lambda=0, validation_loader=None, get_training_details=False,
              download_path=None, fold=0, final_epoch=False):
    """Train a binary classification DNN for one epoch."""
    model.train()
    train_loss = 0
    actual, prediction, all_data, predicted_probability = [], [], [], []

    for batch in data_loader:
        data, target = batch
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()

        out = model(data)
        out = out.squeeze(0) if out.shape[0] == 1 else out.squeeze()
        prob = torch.sigmoid(out)

        loss = loss_criteria(out, target)
        if l1_lambda != 0:
            loss = loss + L1_loss(model) * l1_lambda

        train_loss += loss.cpu().item()
        predicted = (out.data.sigmoid() > CLASSIFICATION_THRESHOLD).long()
        actual.extend(target.cpu())
        prediction.extend(predicted.cpu())
        all_data.append(data.cpu())
        predicted_probability.extend(prob.cpu())

        loss.backward()
        optimizer.step()

    actuals, predictions, all_data_np, pred_probs = _collect_predictions(
        actual, prediction, predicted_probability, all_data)

    if final_epoch:
        acc, roc_auc, f1, prec, rec, roc_result = calc_metrics_with_ci(
            actuals, predictions, pred_probs, metrics=METRICS_LIST,
            download_path=download_path, fold=fold, training=True)
    else:
        acc, roc_auc, f1, prec, rec, roc_result = calc_metrics_updated(
            actuals, predictions, pred_probs, metrics=METRICS_LIST)

    balanced_accuracy = acc * 100
    avg_loss = train_loss / max(len(data_loader), 1)

    print(f"Lr={optimizer.param_groups[0]['lr']:.4f}")
    print(f"Training set: Average loss: {avg_loss:.6f}, "
          f"Balanced Accuracy: {balanced_accuracy:.2f}%, F1: {f1:.2f}")

    if not get_training_details and validation_loader is None:
        return avg_loss, balanced_accuracy

    overall_cm = confusion_matrix(actual, prediction)
    calculate_metrics(overall_cm)

    result_df = pd.DataFrame(
        np.concatenate([all_data_np, actuals.reshape(-1, 1),
                        predictions.reshape(-1, 1), pred_probs.reshape(-1, 1)], axis=1),
        columns=(features or []) + ['Actual', 'Prediction', 'predicted_probability'])

    if validation_loader is not None:
        val_loss, val_bacc, val_steps, validation_df = validate_dnn(
            model, validation_loader, loss_criteria, features)
        if get_training_details:
            return (avg_loss, balanced_accuracy, val_loss, val_steps,
                    np.mean(acc), np.mean(roc_auc), np.mean(f1),
                    np.mean(prec), np.mean(rec), roc_result, overall_cm,
                    actuals, pred_probs, result_df, validation_df)
        return avg_loss, balanced_accuracy, val_loss, val_steps

    if get_training_details:
        return (avg_loss, balanced_accuracy, np.mean(acc), np.mean(roc_auc),
                np.mean(f1), np.mean(prec), np.mean(rec), roc_result,
                overall_cm, actuals, pred_probs, result_df)

    return avg_loss, balanced_accuracy


def validate_dnn(model, data_loader, loss_criteria, features):
    """Evaluate the model on the validation split without gradient computation."""
    model.eval()
    total_val_loss = 0
    val_steps = 0
    actual, prediction, all_data, predicted_probability = [], [], [], []

    with torch.no_grad():
        for batch in data_loader:
            data, target = batch
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = model(data)
            out = out.squeeze(0) if out.shape[0] == 1 else out.squeeze()

            prob = torch.sigmoid(out)
            val_loss = loss_criteria(out, target)
            total_val_loss += val_loss.cpu().item()
            val_steps += 1

            predicted = (out.data.sigmoid() > CLASSIFICATION_THRESHOLD).long()
            all_data.append(data.cpu())
            actual.extend(target.cpu())
            prediction.extend(predicted.cpu())
            predicted_probability.extend(prob.cpu())

    actuals = np.array([t.item() for t in actual])
    predictions = np.array([t.item() for t in prediction])
    all_data_np = torch.cat(all_data, dim=0).numpy()
    pred_probs = np.array([p.detach().numpy() for p in predicted_probability])

    balanced_accuracy = balanced_accuracy_score(actuals, predictions) * 100
    f1 = f1_score(actuals, predictions)

    result_df = pd.DataFrame(
        np.concatenate([all_data_np, actuals.reshape(-1, 1),
                        predictions.reshape(-1, 1), pred_probs.reshape(-1, 1)], axis=1),
        columns=features + ['Actual', 'Prediction', 'predicted_probability'])

    avg_loss = total_val_loss / max(len(data_loader), 1)
    print(f"Validation set: Average loss: {avg_loss:.6f}, "
          f"Balanced Accuracy: {balanced_accuracy:.2f}%, F1: {f1:.2f}")

    return avg_loss, balanced_accuracy, val_steps, result_df


def test_dnn(model, data_loader, loss_criteria, features, final=False,
             msia=True, model_size='large', get_all_layers_output=True,
             download_path=None, fold=0, save=True):
    """Evaluate a binary classification DNN on a held-out split."""
    model.eval()
    actual, prediction, all_data, predicted_probability = [], [], [], []
    test_loss = 0

    output_layer_dict = None
    if get_all_layers_output:
        output_layer_dict = {key: [] for key in LAYER_OUTPUTS[model_size]}

    for batch_index, tensor in enumerate(data_loader):
        with torch.no_grad():
            data, target = tensor
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = model(data)

            if get_all_layers_output:
                _extract_layer_outputs(model, data, output_layer_dict)

            prob = torch.sigmoid(out)
            out = out.squeeze(0) if out.shape[0] == 1 else out.squeeze()
            loss = loss_criteria(out, target)
            test_loss += loss.cpu().item()

            predicted = (out.data.sigmoid() > CLASSIFICATION_THRESHOLD).long()
            all_data.append(data.cpu())
            actual.extend(target.cpu())
            prediction.extend(predicted.cpu())
            predicted_probability.extend(prob.cpu())

    avg_loss = test_loss / (batch_index + 1)

    output_layer_dict_final = (_convert_tensors_to_dataframe(output_layer_dict)
                               if get_all_layers_output else {})

    actuals = np.array(actual)
    predictions = np.array(prediction)
    all_data_np = torch.cat(all_data, dim=0).numpy()
    pred_probs = np.array([p.detach().numpy() for p in predicted_probability])

    acc, roc_auc, f1, prec, rec, roc_result = calc_metrics_with_ci(
        actuals, predictions, pred_probs, metrics=METRICS_LIST,
        download_path=download_path, fold=fold, training=False, save=save)

    result_df = pd.DataFrame(
        np.concatenate([all_data_np, actuals.reshape(-1, 1),
                        predictions.reshape(-1, 1), pred_probs.reshape(-1, 1)], axis=1),
        columns=features + ['Actual', 'Prediction', 'predicted_probability'])

    balanced_accuracy = acc * 100
    dataset_name = "Malaysia" if msia else "India"
    print(f"Validation set ({dataset_name}): Average loss: {avg_loss:.6f}, "
          f"Balanced Accuracy: {balanced_accuracy:.2f}%, F1: {f1:.2f}")

    if final:
        overall_cm = confusion_matrix(actual, prediction)
        _print_final_metrics(acc, roc_auc, f1, prec, rec, overall_cm)
        return (avg_loss, balanced_accuracy, np.mean(acc), np.mean(roc_auc),
                np.mean(f1), np.mean(prec), np.mean(rec), roc_result,
                overall_cm, result_df, output_layer_dict_final,
                actuals, pred_probs)

    return (avg_loss, balanced_accuracy, np.mean(acc), np.mean(roc_auc),
            np.mean(f1), np.mean(prec), np.mean(rec), result_df)
