"""Hyperparameter search construction."""

from __future__ import annotations

from sga import config
from sga.models.estimators import build_sklearn_search
from sga.models.hyperparameters import classifier_catboost_hyperparameters


def test_grid_searches_use_the_configured_inner_folds_and_auroc():
    for model_type in ("lr", "rf", "svc"):
        search, is_search = build_sklearn_search(model_type)
        assert is_search
        assert search.cv == config.GRID_SEARCH_CV_FOLDS
        assert search.scoring == "roc_auc"


def test_stacking_is_not_grid_searched():
    stacking, is_search = build_sklearn_search("stacking")
    assert not is_search


def test_catboost_grid_uses_the_configured_boosting_types():
    grid = classifier_catboost_hyperparameters()
    assert grid["boosting_type"] == list(config.CATBOOST_BOOSTING_TYPES)
