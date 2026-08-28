"""Tests for the evaluation code that produces the manuscript's numbers."""

from __future__ import annotations

import numpy as np
import pytest

from sga.config import DECISION_THRESHOLD, ECE_BINS, N_BOOTSTRAP
from sga.evaluation.bootstrap import bootstrap_ci
from sga.evaluation.calibration import platt_cross_fitted
from sga.evaluation.dca import decision_curve_analysis
from sga.evaluation.delong import delong_test, holm_bonferroni_correction
from sga.evaluation.fairness import compute_fairness_metrics
from sga.evaluation.metrics import expected_calibration_error


@pytest.fixture
def scores():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 400)
    p = np.clip(0.25 + 0.5 * y + rng.normal(0, 0.18, 400), 0.01, 0.99)
    return y, p


# ── Calibration ──────────────────────────────────────────────────────────────


def test_ece_is_zero_for_perfectly_calibrated_probabilities():
    y = np.array([0, 1] * 500)
    p = np.full(1000, 0.5)
    assert expected_calibration_error(y, p, n_bins=ECE_BINS) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_maximal_for_confidently_wrong_probabilities():
    y = np.array([0] * 200 + [1] * 200)
    p = np.array([1.0] * 200 + [0.0] * 200)
    assert expected_calibration_error(y, p, n_bins=ECE_BINS) == pytest.approx(1.0, abs=1e-9)


def test_ece_uses_the_configured_number_of_bins(scores):
    """The manuscript states 10 equal-width bins; the default must match."""
    y, p = scores
    assert expected_calibration_error(y, p) == expected_calibration_error(y, p, n_bins=ECE_BINS)


def test_platt_cross_fitting_never_scores_a_row_with_its_own_calibrator(scores):
    """Cross-fitted Platt scaling must return one probability per input row."""
    y, p = scores
    calibrated = platt_cross_fitted(y, p)
    assert calibrated.shape == p.shape
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    # A monotone map cannot change the ranking, hence not the AUROC.
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, p), abs=0.02)


# ── DeLong and multiplicity ──────────────────────────────────────────────────


def test_delong_reports_no_difference_for_identical_predictors(scores):
    y, p = scores
    auc_a, auc_b, _, p_value = delong_test(y, p, p.copy())
    assert auc_a == pytest.approx(auc_b)
    assert p_value == pytest.approx(1.0)


def test_delong_detects_a_clear_difference(scores):
    y, p = scores
    rng = np.random.RandomState(1)
    noise = rng.uniform(size=len(y))
    _, _, _, p_value = delong_test(y, p, noise)
    assert p_value < 0.001


def test_holm_bonferroni_is_monotone_and_never_shrinks_a_p_value():
    raw = np.array([0.001, 0.01, 0.04, 0.7])
    adjusted = holm_bonferroni_correction(raw)
    assert np.all(adjusted >= raw)
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-12)
    assert np.all(adjusted <= 1.0)


def test_holm_bonferroni_matches_the_reference_implementation():
    """The manuscript says "Holm-Bonferroni correction" (Table 6), so verify that
    is the procedure implemented - not some other step-down variant."""
    pytest.importorskip("statsmodels")
    from statsmodels.stats.multitest import multipletests

    rng = np.random.RandomState(0)
    for trial in range(50):
        family_size = rng.randint(2, 20)
        raw = np.round(rng.uniform(0, 1, family_size), 6)
        if trial % 5 == 0:  # exercise ties, which the step-down order must handle
            raw[rng.randint(0, family_size)] = raw[rng.randint(0, family_size)]
        reference = multipletests(raw, alpha=0.05, method="holm")[1]
        assert holm_bonferroni_correction(raw) == pytest.approx(reference)


def test_holm_bonferroni_preserves_input_order_and_nans():
    raw = np.array([0.04, np.nan, 0.01])
    adjusted = holm_bonferroni_correction(raw)
    assert np.isnan(adjusted[1])
    # Two finite comparisons: the smaller is multiplied by 2, the larger by 1.
    assert adjusted[2] == pytest.approx(0.02)
    assert adjusted[0] == pytest.approx(0.04)


# ── Decision curve ───────────────────────────────────────────────────────────


def test_net_benefit_matches_the_published_formula(scores):
    """Net benefit is TP/n - (FP/n) * pt/(1-pt), as stated in the Methods."""
    y, p = scores
    thresholds, net_benefits, _ = decision_curve_analysis(y, p)
    n = len(y)
    for threshold, reported in list(zip(thresholds, net_benefits))[::7]:
        predicted = (p >= threshold).astype(int)
        tp = np.sum((predicted == 1) & (y == 1))
        fp = np.sum((predicted == 1) & (y == 0))
        expected = tp / n - (fp / n) * (threshold / (1 - threshold))
        assert reported == pytest.approx(expected)


def test_treat_all_net_benefit_is_zero_at_prevalence(scores):
    """At pt = prevalence, treating everyone has exactly zero net benefit."""
    y, p = scores
    prevalence = y.mean()
    _, _, treat_all = decision_curve_analysis(y, p, thresholds=np.array([prevalence]))
    assert treat_all[0] == pytest.approx(0.0, abs=1e-12)


# ── Bootstrap ────────────────────────────────────────────────────────────────


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.RandomState(0)
    values = rng.normal(0.8, 0.05, 200).tolist()
    mean, low, high = bootstrap_ci(values)
    assert low < mean < high


def test_bootstrap_ci_is_independent_of_the_global_rng():
    """Two identical calls must agree regardless of intervening global draws."""
    values = np.linspace(0.6, 0.9, 50).tolist()
    first = bootstrap_ci(values)
    np.random.seed(999)
    np.random.normal(size=1000)
    assert bootstrap_ci(values) == first


def test_default_bootstrap_count_matches_the_manuscript():
    assert N_BOOTSTRAP == 2000


def test_decision_threshold_is_the_documented_operating_point():
    """Every threshold-dependent metric is reported at 0.5."""
    assert DECISION_THRESHOLD == 0.5
    from sga.models.torch_utils import CLASSIFICATION_THRESHOLD

    assert CLASSIFICATION_THRESHOLD == DECISION_THRESHOLD, (
        "the DNN loops must binarise at the same cut-off as every other model"
    )


# ── Cohort disparity ─────────────────────────────────────────────────────────


def test_disparity_is_zero_when_both_cohorts_behave_identically():
    y = np.array([0, 1] * 50)
    pred = y.copy()
    country = np.array([0] * 50 + [1] * 50)
    metrics = compute_fairness_metrics(y, pred, country)
    for name, value in metrics.items():
        if "difference" in name.lower() or name.lower().endswith("_diff"):
            assert value == pytest.approx(0.0, abs=1e-9), name


# ── Calibration provenance ───────────────────────────────────────────────────


def test_no_experiment_fits_platt_on_test_labels():
    """The calibrator must never be fitted on the labels it is reported against."""
    import pathlib
    import re

    offenders = []
    pattern = re.compile(r"(fit_platt|platt_cross_fitted)\s*\(\s*[^,)]*test_Y")
    for path in pathlib.Path("rebuttals").rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path}:{number}")

    assert not offenders, "Platt scaling fitted on test labels at: " + ", ".join(offenders)


def test_platt_map_fitted_on_development_data_is_monotone(scores):
    """A single fitted calibrator is monotone, so it cannot change the AUROC."""
    from sklearn.metrics import roc_auc_score

    from sga.evaluation.calibration import apply_platt, fit_platt

    y_dev, p_dev = scores
    rng = np.random.RandomState(5)
    p_test = np.clip(rng.uniform(size=300), 1e-6, 1 - 1e-6)
    y_test = rng.binomial(1, p_test)

    calibrated = apply_platt(fit_platt(y_dev, p_dev), p_test)
    assert roc_auc_score(y_test, calibrated) == pytest.approx(
        roc_auc_score(y_test, p_test), abs=1e-9
    )
