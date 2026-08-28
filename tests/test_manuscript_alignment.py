"""Assertions that tie the code's constants to statements in the manuscript.

Each test names the sentence it is enforcing. When the paper changes, one of these
fails and points at the code that has to move with it, so the two cannot drift
apart unnoticed.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from sga import config


# ── Methods: leakage-safe cross-validation ───────────────────────────────────


def test_five_patient_grouped_folds_with_fold_four_held_out():
    """"partitioned by patient into five folds ... fold 4 was held out"."""
    assert config.N_FOLDS_TOTAL == 5
    assert config.N_FOLDS_CV == 4
    assert config.EXTERNAL_TEST_FOLD == 4
    assert config.N_FOLDS_CV + 1 == config.N_FOLDS_TOTAL


def test_global_seed_is_the_one_the_figures_declare():
    """"All experiments run under the global seed of 123" (Figure 7 caption)."""
    assert config.SEED == 123


def test_ten_common_features():
    """The Methods list exactly ten features measured in both cohorts."""
    assert len(config.COMMON_FEATURES) == 10
    assert len(set(config.COMMON_FEATURES)) == 10


def test_only_af_and_ute_ari_are_retained_cross_domain_features():
    """"only amniotic fluid condition (af) and uterine artery resistance index
    (ute_ari) satisfied these criteria and were retained"."""
    assert sorted(config.HARMONIZED_SELECTED_FEATURES) == ["af", "ute_ari"]


def test_imputation_retention_thresholds_match_the_published_criteria():
    """Binary AUROC > 0.80 with F1 > 0.50; multiclass F1 > 0.80 with AUROC > 0.50;
    continuous R^2 > 0.80."""
    assert config.BINARY_AUROC_THRESHOLD == 0.80
    assert config.BINARY_F1_THRESHOLD == 0.50
    assert config.MULTICLASS_F1_THRESHOLD == 0.80
    assert config.MULTICLASS_AUROC_THRESHOLD == 0.50
    assert config.REGRESSION_R2_THRESHOLD == 0.80


def test_constant_features_are_excluded_not_silently_retained():
    """Appendix Table S2: binary features with one observed response were excluded."""
    from sga.imputation.apply import select_features_with_threshold

    selected, removed = select_features_with_threshold(
        list(config.CONSTANT_ZERO_FEATURES), selection_type="binary"
    )
    assert selected == []
    assert sorted(removed) == sorted(config.CONSTANT_ZERO_FEATURES)


# ── Model evaluation ─────────────────────────────────────────────────────────


def test_ece_uses_ten_equal_width_bins():
    """"Expected calibration error was calculated using 10 equal-width bins"."""
    assert config.ECE_BINS == 10


def test_primary_threshold_dependent_results_use_the_default_cut_off():
    """"The primary threshold-dependent results use the fixed 0.50 cut-off"."""
    assert config.DECISION_THRESHOLD == 0.5


def test_screening_threshold_lies_inside_the_clinically_relevant_range():
    """"chosen a priori as representative of the 5-20% clinically relevant range"."""
    low, high = config.DCA_CLINICAL_RANGE
    assert (low, high) == (0.05, 0.20)
    assert low <= config.SCREENING_THRESHOLD <= high


def test_threshold_grid_matches_appendix_tables_s4_and_s5():
    """Both tables run from 0.10 to 0.90 in steps of 0.05."""
    grid = config.THRESHOLD_GRID
    assert grid[0] == 0.10
    assert grid[-1] == 0.90
    assert len(grid) == 17
    assert np.allclose(np.diff(grid), 0.05)


def test_bootstrap_uses_two_thousand_iterations():
    """"computed using the bootstrap method with 2,000 iterations"."""
    assert config.N_BOOTSTRAP == 2000
    assert config.ALPHA == 0.05


def test_proportions_use_score_intervals_and_only_the_rest_are_bootstrapped():
    """"Confidence intervals for proportions ... Wilson score method. The
    intervals for differences between two independent proportions ... Newcombe's
    hybrid score method." Bootstrap is for AUROC, AUPRC, ECE and Brier."""
    assert config.CI_METHODS["proportion"] == "Wilson score"
    assert config.CI_METHODS["proportion_difference"] == "Newcombe hybrid score"
    for metric in ("auroc", "auprc", "ece", "brier"):
        assert config.CI_METHODS[metric] == "percentile bootstrap"


def test_six_classifiers_are_named_as_the_tables_name_them():
    assert set(config.MODEL_DISPLAY_NAMES) == {
        "catboost", "rf", "lr", "svc", "stacking", "dnn"
    }
    assert config.MODEL_DISPLAY_NAMES["dnn"] == "Neural Network"


# ── Calibration provenance ───────────────────────────────────────────────────

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_PLATT_ON_TEST = re.compile(r"(fit_platt|platt_cross_fitted)\s*\(\s*[^,)]*test_Y")


@pytest.mark.parametrize("package", ["sga", "rebuttals", "scripts"])
def test_no_module_fits_platt_on_the_labels_it_reports(package):
    """"The calibrator was never fitted using the labels of the samples on which
    the final metrics were reported"."""
    offenders = []
    root = _PROJECT_ROOT / package
    if not root.exists():
        pytest.skip(f"{package} not present")
    for path in root.rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if _PLATT_ON_TEST.search(line):
                offenders.append(f"{path.relative_to(_PROJECT_ROOT)}:{number}")
    assert not offenders, "Platt scaling fitted on test labels at: " + ", ".join(offenders)


def test_catboost_training_never_passes_the_test_fold_as_an_eval_set():
    """CatBoost turns on ``use_best_model`` whenever an eval_set is supplied, which
    would select the stopping iteration on the reported rows."""
    source = (_PROJECT_ROOT / "sga" / "pipeline" / "train_unified.py").read_text()
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "eval_set" in line and "#" not in line.split("eval_set")[0]
    ]
    assert not offenders, "test fold reached CatBoost as an eval_set: " + "; ".join(offenders)


# ── Calibration base rate ────────────────────────────────────────────────────


def test_calibration_is_fitted_on_authentic_held_out_predictions():
    """"Platt scaling was fitted by cross-fitting on held-out validation
    predictions."

    Held-out validation rows are authentic: SMOTENC is applied to training rows
    only. Fitting the calibrator on the assembled (oversampled) training matrix
    instead would anchor the Platt map to a 50% base rate and apply it to a fold at
    the observed ~21.6% prevalence, where it cannot reduce the calibration error.
    """
    source = (_PROJECT_ROOT / "sga" / "pipeline" / "external_fold.py").read_text()
    assert "def development_out_of_fold_predictions" in source
    # The out-of-fold predictions must come from the fold builder, not from a
    # cross-validation of the already-resampled training matrix. (The docstring
    # names `cross_val_predict` to explain why it is not used, so look for a call.)
    assert "cross_val_predict(" not in source, (
        "development predictions are being taken over the SMOTENC-resampled "
        "training matrix again; they must come from each fold's held-out rows"
    )
    assert "exclude_external_fold=True" in source


def test_platt_map_moves_probabilities_towards_the_base_rate():
    """A calibrator fitted at the right base rate must reduce the ECE."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from sga.evaluation.calibration import apply_platt, fit_platt
    from sga.evaluation.metrics import expected_calibration_error

    rng = np.random.RandomState(0)
    # Over-confident scores at a 20% base rate, as an uncalibrated model produces.
    y_dev = rng.binomial(1, 0.2, 4000)
    p_dev = np.clip(0.5 + 0.35 * (2 * y_dev - 1) + rng.normal(0, 0.15, 4000), 0.01, 0.99)
    y_test = rng.binomial(1, 0.2, 1000)
    p_test = np.clip(0.5 + 0.35 * (2 * y_test - 1) + rng.normal(0, 0.15, 1000), 0.01, 0.99)

    calibrated = apply_platt(fit_platt(y_dev, p_dev), p_test)
    assert expected_calibration_error(y_test, calibrated) < expected_calibration_error(
        y_test, p_test
    )
    assert abs(calibrated.mean() - y_test.mean()) < 0.05
    assert isinstance(fit_platt(y_dev, p_dev), LogisticRegression)


def test_the_external_partition_is_scored_by_a_development_fold_model():
    """"the best model, judging from the highest AUROC based on the validation
    sets, was used to evaluate on the testing data" (Methods 2.3.4)."""
    source = (_PROJECT_ROOT / "sga" / "pipeline" / "external_fold.py").read_text()
    assert "validation_auroc" in source and "selected_fold" in source
    assert "external_test_fold=external_test_fold" in source, (
        "fold 4 must be scored through prepare_fold's external_test_fold path, so "
        "the selected fold-model's own preprocessing carries onto it"
    )


def test_within_feature_imputation_happens_inside_the_fold():
    """Methods step (2): "iterative imputation fitted on the training partition"."""
    splits = (_PROJECT_ROOT / "sga" / "data" / "splits.py").read_text()
    dataset = (_PROJECT_ROOT / "sga" / "pipeline" / "dataset.py").read_text()
    assert "impute_data=False" in splits, (
        "data preparation must leave the gaps for the fold builder to fill"
    )
    assert "def impute_within_feature" in dataset
    assert "fit_iterative_imputer" in dataset
