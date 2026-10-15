"""Grid-searched estimators shared by the unified and country-baseline pipelines.

The search settings are those of appendix Table S11: AUROC as the selection metric,
``GRID_SEARCH_CV_FOLDS`` inner folds for the six-classifier comparison, CatBoost
restricted to ``CATBOOST_BOOSTING_TYPES`` with early stopping on the fold's held-out
validation rows, and ``LR_REFIT_CV_FOLDS`` inner folds for the Logistic Regression
refitted for the calibration, threshold and fairness analyses.
"""

from __future__ import annotations

from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sga.config import (
    CATBOOST_EARLY_STOPPING_ROUNDS,
    GRID_SEARCH_CV_FOLDS,
    LR_REFIT_CV_FOLDS,
    SEED,
)
from sga.models.ensemble import build_stacking_classifier
from sga.models.hyperparameters import (
    classifier_catboost_hyperparameters,
    lr_hyperparameters,
    rf_hyperparameters,
    svc_hyperparameters,
)

LR_MAX_ITER = 1000

SKLEARN_SEARCHES = {
    "rf": (RandomForestClassifier, rf_hyperparameters),
    "lr": (LogisticRegression, lr_hyperparameters),
    "svc": (SVC, svc_hyperparameters),
}


def train_lr(train_X, train_Y, seed=SEED, cv=LR_REFIT_CV_FOLDS):
    """Fit the Logistic Regression with its Table S11 grid search."""
    base = LogisticRegression(max_iter=LR_MAX_ITER, class_weight="balanced", random_state=seed)
    search = GridSearchCV(base, lr_hyperparameters(), scoring="roc_auc", cv=cv)
    search.fit(train_X, train_Y)
    return search.best_estimator_


def build_sklearn_search(model_type, cv=GRID_SEARCH_CV_FOLDS):
    """``(estimator, is_grid_search)`` for LR, RF, SVC or the fixed Stacking ensemble."""
    if model_type == "stacking":
        return build_stacking_classifier(), False
    if model_type not in SKLEARN_SEARCHES:
        raise ValueError(
            f"Invalid model type {model_type!r}; choose from {sorted(SKLEARN_SEARCHES) + ['stacking']}"
        )
    model_class, grid = SKLEARN_SEARCHES[model_type]
    base = model_class(probability=True) if model_type == "svc" else model_class()
    return GridSearchCV(base, grid(), cv=cv, scoring="roc_auc", n_jobs=-1), True


def train_catboost_classifier(
    train_X,
    train_Y,
    validation_X,
    validation_Y,
    categorical_features,
    task_type="CPU",
    cv=GRID_SEARCH_CV_FOLDS,
    early_stopping_rounds=CATBOOST_EARLY_STOPPING_ROUNDS,
):
    """Grid-search a CatBoost classifier and refit it with the selected parameters.

    Returns ``(model, best_params)``. Every fit stops early on the validation rows.
    """
    search = GridSearchCV(
        estimator=CatBoostClassifier(class_names=[0, 1], task_type=task_type),
        param_grid=classifier_catboost_hyperparameters(),
        scoring="roc_auc",
        cv=cv,
        n_jobs=1 if task_type == "GPU" else -1,
    )
    search.fit(
        train_X,
        train_Y,
        eval_set=(validation_X, validation_Y),
        early_stopping_rounds=early_stopping_rounds,
        verbose=100,
    )
    model = CatBoostClassifier(**search.best_params_, class_names=[0, 1], task_type=task_type)
    model.fit(
        Pool(data=train_X, label=train_Y, cat_features=categorical_features),
        eval_set=Pool(data=validation_X, label=validation_Y, cat_features=categorical_features),
        early_stopping_rounds=early_stopping_rounds,
        verbose=100,
    )
    return model, search.best_params_
