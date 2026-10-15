"""PyTorch helpers shared by the DNN training and evaluation code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.utils.data as td

from sga.config import DECISION_THRESHOLD, SEED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Operating point for binarising the network's sigmoid output.
CLASSIFICATION_THRESHOLD = DECISION_THRESHOLD


def _create_dataloader(X, Y, batch_size, shuffle=True, drop_last=True,
                       balanced_sampling=False):
    """Create a PyTorch DataLoader from NumPy arrays."""
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    dataset = td.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))

    if balanced_sampling:
        try:
            from torchsampler import ImbalancedDatasetSampler
            return td.DataLoader(dataset, sampler=ImbalancedDatasetSampler(dataset),
                                 batch_size=batch_size, drop_last=drop_last)
        except ImportError:
            print("Warning: torchsampler not available, falling back to standard sampling")

    return td.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         drop_last=drop_last)


def convert_to_tensor_dataloader(train_X=None, train_Y=None, test_X=None,
                                 test_Y=None, batch_size=None,
                                 validation_X=None, validation_Y=None,
                                 pytorch_balance_sampling=False):
    """Convert NumPy arrays to PyTorch DataLoaders."""
    train_loader = None
    test_loader = None
    validation_loader = None

    if train_X is not None and train_Y is not None:
        train_loader = _create_dataloader(
            train_X, train_Y, batch_size, shuffle=True,
            balanced_sampling=pytorch_balance_sampling)

    if test_X is not None and test_Y is not None:
        test_loader = _create_dataloader(
            test_X, test_Y, batch_size, shuffle=False, drop_last=False)

    if validation_X is not None and validation_Y is not None:
        validation_loader = _create_dataloader(
            validation_X, validation_Y, batch_size, shuffle=False, drop_last=False)
        return train_loader, test_loader, validation_loader

    return train_loader, test_loader


def adjust_learning_rate(optimizer, lr_scheduler, min_learning_rate):
    """Step the scheduler, then clamp every parameter group to a minimum rate."""
    lr_scheduler.step()
    for param_group in optimizer.param_groups:
        param_group["lr"] = max(param_group["lr"], min_learning_rate)


def compute_permutation_importance(model, data, target, metric, feature_names,
                                   n_repeats=10, seed=SEED):
    """Compute permutation feature importance for a trained DNN."""
    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    if hasattr(target, "cpu"):
        target = target.cpu().numpy()

    predictions = model(torch.tensor(data).to(DEVICE)).detach().cpu().numpy()
    baseline = metric(target, predictions)

    rng = np.random.RandomState(seed)
    importances = []
    for col in range(data.shape[1]):
        scores = []
        for _ in range(n_repeats):
            permuted = data.copy()
            rng.shuffle(permuted[:, col])
            perm_preds = model(torch.tensor(permuted).to(DEVICE)).detach().cpu().numpy()
            scores.append(baseline - metric(target, perm_preds))
        importances.append(np.mean(scores))

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False)
