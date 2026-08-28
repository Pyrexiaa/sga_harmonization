"""Hyperparameter search grids for the classical and boosted-tree models."""

from __future__ import annotations

import numpy as np


def rf_hyperparameters():
    """Grid for the Random Forest classifier (binary SGA prediction)."""
    n_estimators = [10, 100, 200]
    max_depth = [10, 50, 100, None]
    min_samples_split = [2, 5]
    min_samples_leaf = [1, 2]
    class_weight = ['balanced']

    grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'class_weight': class_weight,
    }

    return grid


def rf_regression_hyperparameters():
    """Grid for the Random Forest regressor used by the continuous imputers."""
    grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 50, 100],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        # 'auto' was removed from RandomForestRegressor in scikit-learn 1.3; 1.0 is its
        # documented replacement (use every feature).
        'max_features': [1.0, 'sqrt', 'log2'],
    }

    return grid


def lr_hyperparameters():
    """Grid for the Logistic Regression classifier (binary SGA prediction)."""
    grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "class_weight": ["balanced"],
        "max_iter": [200, 500],
    }

    return grid


def lr_multiclass_hyperparameters():
    """Grid for the multiclass Logistic Regression cross-domain imputer."""
    grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs", "saga"],
        # `multi_class` was deprecated in scikit-learn 1.5:
        "class_weight": ["balanced"],
        "max_iter": [200, 500],
    }

    return grid


def ridge_hyperparameters():
    """Grid for the Ridge regressor used by the continuous imputers."""
    grid = {
        "alpha": [0.01, 0.1, 1, 10],
        "solver": ["auto", "svd", "cholesky", "lsqr"],
        "fit_intercept": [True, False],
    }
    return grid


def svc_hyperparameters():
    """Grid for the Support Vector Classifier (binary SGA prediction)."""
    grid = {
        'C': [1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale'],
        'class_weight': ['balanced'],
    }

    return grid


def svr_hyperparameters():
    """Grid for the Support Vector Regressor used by the continuous imputers."""
    grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.1, 0.2, 0.5],
        'shrinking': [True, False],
    }

    return grid


def stacking_hyperparameters():
    """Grid for the Stacking Classifier's Random Forest and SVC base learners."""
    grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [100],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1],
        'rf__class_weight': ['balanced'],
        'svc__C': [10],
        'svc__kernel': ['rbf'],
        'svc__gamma': ['scale'],
        'svc__class_weight': ['balanced'],
    }

    return grid


def catboost_hyperparameters():
    """Grid for the binary CatBoost classifier (SGA prediction)."""
    grid = {
        'iterations': [500],
        'depth': [4, 5, 6],
        'loss_function': ['Logloss'],
        'eval_metric': ['TotalF1'],
        'l2_leaf_reg': np.logspace(-3, 2, 5),
        'leaf_estimation_iterations': [10],
        'auto_class_weights': ['Balanced'],
        'learning_rate': [0.01, 0.03, 0.1],
        'boosting_type': ['Ordered', 'Plain'],
    }

    return grid


def multiclass_catboost_hyperparameters():
    """Grid for the multiclass CatBoost cross-domain imputer."""
    grid = {
        'iterations': [500],
        'depth': [4, 5, 6],
        'loss_function': ['MultiClass'],
        'eval_metric': ['TotalF1'],
        'l2_leaf_reg': np.logspace(-3, 2, 5),
        'leaf_estimation_iterations': [10],
        'auto_class_weights': ['Balanced'],
        'learning_rate': [0.01, 0.03, 0.1],
        'boosting_type': ['Ordered', 'Plain'],
    }

    return grid


def regression_catboost_hyperparameters():
    """Grid for the CatBoost regressor (generic continuous targets)."""
    grid = {
        'iterations': [500],
        'depth': [4, 5, 6],
        'loss_function': ['RMSE'],
        'eval_metric': ['RMSE'],
        'l2_leaf_reg': np.logspace(-3, 2, 5),
        'leaf_estimation_iterations': [10],
        'learning_rate': [0.01, 0.03, 0.1],
        'boosting_type': ['Ordered', 'Plain'],
    }

    return grid


def regression_impute_catboost_hyperparameters():
    """Grid for the CatBoost continuous cross-domain imputers."""
    grid = {
        'iterations': [2000],
        'depth': [3, 4, 5, 6],
        'loss_function': ['RMSE'],
        'eval_metric': ['RMSE'],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 10, 30],
        'min_data_in_leaf': [5, 10, 20],
        'random_strength': [1, 5, 10],
        'bagging_temperature': [0, 0.5, 1],
        'boosting_type': ['Ordered'],
        'leaf_estimation_iterations': [10],
    }
    return grid


def improved_regression_catboost_hyperparameters():
    """Grid for the CatBoost continuous imputers on weak-signal targets."""
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
