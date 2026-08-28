"""Plain scikit-learn estimators used directly by the analysis scripts."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sga.config import SEED
from sga.models.hyperparameters import lr_hyperparameters

LR_GRID_SEARCH_CV_FOLDS = 5
LR_MAX_ITER = 1000


def train_lr(train_X, train_Y, seed=SEED):
    """Fit the paper's Logistic Regression with the standard grid search."""
    base = LogisticRegression(max_iter=LR_MAX_ITER, class_weight="balanced", random_state=seed)
    search = GridSearchCV(
        base, lr_hyperparameters(), scoring="roc_auc", cv=LR_GRID_SEARCH_CV_FOLDS
    )
    search.fit(train_X, train_Y)
    return search.best_estimator_
