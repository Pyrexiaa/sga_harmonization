"""Hyperparameter search grids (appendix Table S11 for the six classifiers)."""

from __future__ import annotations

import numpy as np

from sga.config import CATBOOST_BOOSTING_TYPES


def rf_hyperparameters():
    """Grid for the Random Forest classifier."""
    return {
        "n_estimators": [10, 100, 200],
        "max_depth": [10, 50, 100, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": ["balanced"],
    }


def lr_hyperparameters():
    """Grid for the Logistic Regression classifier."""
    return {
        "C": [0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "class_weight": ["balanced"],
        "max_iter": [200, 500],
    }


def svc_hyperparameters():
    """Grid for the Support Vector Classifier."""
    return {
        "C": [1, 10],
        "kernel": ["rbf"],
        "gamma": ["scale"],
        "class_weight": ["balanced"],
    }


def catboost_hyperparameters(boosting_types=("Ordered", "Plain")):
    """Grid for a binary CatBoost classifier."""
    return {
        "iterations": [500],
        "depth": [4, 5, 6],
        "loss_function": ["Logloss"],
        "eval_metric": ["TotalF1"],
        "l2_leaf_reg": np.logspace(-3, 2, 5),
        "leaf_estimation_iterations": [10],
        "auto_class_weights": ["Balanced"],
        "learning_rate": [0.01, 0.03, 0.1],
        "boosting_type": list(boosting_types),
    }


def classifier_catboost_hyperparameters():
    """The CatBoost SGA classifier's grid as searched in the reported run."""
    return catboost_hyperparameters(boosting_types=CATBOOST_BOOSTING_TYPES)


def regression_catboost_hyperparameters():
    """Grid for a CatBoost regressor."""
    return {
        "iterations": [500],
        "depth": [4, 5, 6],
        "loss_function": ["RMSE"],
        "eval_metric": ["RMSE"],
        "l2_leaf_reg": np.logspace(-3, 2, 5),
        "leaf_estimation_iterations": [10],
        "learning_rate": [0.01, 0.03, 0.1],
        "boosting_type": ["Ordered", "Plain"],
    }


def improved_regression_catboost_hyperparameters():
    """Grid for the CatBoost continuous cross-domain imputers."""
    return {
        "iterations": [300, 500, 800, 1200],
        "depth": [3, 4, 5, 6],
        "loss_function": ["RMSE"],
        "eval_metric": ["RMSE"],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 10, 30],
        "min_data_in_leaf": [5, 10, 20],
        "random_strength": [1, 5, 10],
        "bagging_temperature": [0, 0.5, 1],
        "boosting_type": ["Ordered"],
    }
