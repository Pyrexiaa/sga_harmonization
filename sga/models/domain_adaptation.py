"""Domain-generalization comparators used as baselines against harmonization."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

# Ridge added to each covariance before the matrix square root, so the inverse stays
# defined when a feature is (near-)constant within a cohort.
CORAL_RIDGE = 1e-6


def coral_align(train_X, train_y, domain_train, source=0, target=1, ridge=CORAL_RIDGE):
    """Align the source cohort's covariance onto the target cohort's."""
    domain_train = np.asarray(domain_train)
    source_mask = domain_train == source
    target_mask = domain_train == target
    source_X = train_X[source_mask]
    target_X = train_X[target_mask]

    n_features = source_X.shape[1]
    source_cov = np.cov(source_X, rowvar=False) + np.eye(n_features) * ridge
    target_cov = np.cov(target_X, rowvar=False) + np.eye(n_features) * ridge
    source_inv_half = np.real(scipy.linalg.sqrtm(np.linalg.inv(source_cov)))
    target_half = np.real(scipy.linalg.sqrtm(target_cov))
    transform = (source_inv_half @ target_half).astype(np.float32)

    source_aligned = (source_X @ transform).astype(np.float32)
    aligned_X = np.concatenate([source_aligned, target_X], axis=0).astype(np.float32)
    aligned_y = np.concatenate([train_y[source_mask], train_y[target_mask]], axis=0)
    return aligned_X, aligned_y, transform


def irm_penalty(logits, y, device):
    """IRM-v1 penalty ``||grad_w loss(w * logits, y)||^2`` at the dummy ``w = 1``."""
    dummy_w = torch.tensor(1.0, device=device, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logits * dummy_w, y.float().unsqueeze(1))
    grad_w = autograd.grad(loss, dummy_w, create_graph=True)[0]
    return grad_w**2


class GradientReversalFunction(autograd.Function):
    """Identity forward, sign-flipped and scaled backward (DANN)."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper around :class:`GradientReversalFunction`."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        """Pass ``x`` through unchanged, reversing its gradient on the way back."""
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainClassifier(nn.Module):
    """Small MLP head that predicts the cohort of origin from a representation."""

    def __init__(self, input_size, hidden_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        """Return the raw domain logit for each row of ``x``."""
        return self.net(x)


__all__ = [
    "CORAL_RIDGE",
    "DomainClassifier",
    "GradientReversalFunction",
    "GradientReversalLayer",
    "coral_align",
    "irm_penalty",
]
