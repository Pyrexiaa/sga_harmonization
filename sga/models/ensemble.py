"""Stacking classifier reported as "Stacking Classifier" in the manuscript."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sga.config import SEED

STACKING_CV_FOLDS = 5


def build_base_estimators():
    """Return the ``(name, estimator)`` pairs stacked by the paper's ensemble."""
    return [
        (
            "rf",
            RandomForestClassifier(
                random_state=SEED,
                n_estimators=100,
                max_depth=100,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
            ),
        ),
        (
            "svc",
            SVC(
                random_state=SEED,
                probability=True,
                C=10,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
            ),
        ),
        ("lr", LogisticRegression()),
    ]


def build_stacking_classifier(cv=STACKING_CV_FOLDS):
    """Build the untrained Stacking Classifier."""
    return StackingClassifier(estimators=build_base_estimators(), cv=cv)
