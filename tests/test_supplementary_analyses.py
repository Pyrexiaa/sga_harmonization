"""EFW-centile comparator, cohort-shift statistics, stratified bootstrap and out-of-fold scoring of incomplete records."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint

from sga.config import COMMON_FEATURES, EXTERNAL_TEST_FOLD, INDIA, LABEL, MALAYSIA
from sga.data.centiles import efw_centile
from sga.data.statistics import (
    covariate_overlap,
    normal_overlap_coefficient,
    partition_summary,
    proportion_smd,
    standardized_mean_difference,
)
from sga.evaluation.bootstrap import _stratified_indices, bootstrap_metric_ci
from sga.evaluation.comparator import (
    centile_risk,
    net_benefit,
    operating_point,
    operating_point_table,
    rule_net_benefit,
    workload_matched_threshold,
)
from sga.evaluation.country import ensemble_predictions_path, save_ensemble_predictions
from sga.evaluation.delong import delong_test, delong_test_unpaired
from sga.pipeline.incomplete_records import (
    COMPLETE,
    INCOMPLETE,
    MIN_EVENTS,
    build_union,
    delong_complete_vs_incomplete,
    imputation_burden,
    score_all_records,
    stratum_metrics,
    table_by_imputation_burden,
    table_by_record_type,
)


# DeLong


def test_unpaired_delong_matches_paired_against_an_uninformative_score():
    """Against a constant score the paired covariance vanishes, so both tests agree."""
    rng = np.random.RandomState(3)
    y = rng.binomial(1, 0.3, 300)
    p = y + rng.normal(scale=1.2, size=len(y))
    auc_paired, half, z_paired, p_paired = delong_test(y, p, np.zeros_like(p))
    auc_unpaired, half_unpaired, z_unpaired, p_unpaired = delong_test_unpaired(y, p, y, np.zeros_like(p))

    assert half == half_unpaired == 0.5
    assert auc_unpaired == pytest.approx(roc_auc_score(y, p))
    assert z_unpaired == pytest.approx(z_paired)
    assert p_unpaired == pytest.approx(p_paired)


def test_unpaired_delong_on_disjoint_samples():
    rng = np.random.RandomState(4)
    y1, y2 = rng.binomial(1, 0.3, 400), rng.binomial(1, 0.3, 400)
    strong = y1 + rng.normal(scale=0.5, size=400)
    weak = y2 + rng.normal(scale=3.0, size=400)
    auc1, auc2, z, p = delong_test_unpaired(y1, strong, y2, weak)
    assert auc1 > auc2 and z > 0 and p < 0.001

    same = delong_test_unpaired(y1, strong, y1, strong)
    assert same[2] == pytest.approx(0.0) and same[3] == pytest.approx(1.0)
    assert np.isnan(delong_test_unpaired(np.zeros(5), np.arange(5), y2, weak)[2])


# EFW centile


@pytest.fixture
def centile_reference():
    return pd.DataFrame(
        {"ga": [140, 141], 2.5: [200.0, 210.0], 10: [240.0, 250.0], 50: [300.0, 310.0],
         90: [360.0, 370.0], 97.5: [400.0, 410.0]}
    )


def test_efw_centile_interpolates_extrapolates_and_keeps_every_row(centile_reference):
    scans = pd.DataFrame(
        {"ga": [140, 140, 140, 140, 141, 200, 140], "efw": [270, 240, 180, 420, 310, 300, np.nan]},
        index=[10, 11, 12, 13, 14, 15, 16],
    )
    out = efw_centile(scans, reference=centile_reference)

    assert out.index.equals(scans.index)
    assert out.loc[10, "efw_centile"] == pytest.approx(30.0)
    assert out.loc[11, "efw_centile"] == pytest.approx(10.0)
    assert out.loc[14, "efw_centile"] == pytest.approx(50.0)
    # Below the 2.5th: slope 7.5 centiles per 40 g, 20 g below -> 2.5 - 3.75, floored.
    assert out.loc[12, "efw_centile"] == pytest.approx(0.01)
    assert out.loc[12, "centile_status"] == "below_table"
    # Above the 97.5th: slope 7.5 per 40 g, 20 g above -> 101.25, capped.
    assert out.loc[13, "efw_centile"] == pytest.approx(99.99)
    assert out.loc[13, "centile_status"] == "above_table"
    assert out.loc[15, "centile_status"] == "no_reference"
    assert out.loc[16, "centile_status"] == "no_reference"
    # The rule is strict: exactly the 10th centile is not flagged.
    assert out["below_rule"].tolist() == [0, 0, 1, 0, 0, 0, 0]


def test_extrapolation_below_table_is_monotone(centile_reference):
    scans = pd.DataFrame({"ga": [140, 140, 140], "efw": [199, 195, 190]})
    centiles = efw_centile(scans, reference=centile_reference)["efw_centile"].to_numpy()
    assert np.all(np.diff(centiles) < 0) and np.all(centiles < 2.5)


# Table S9


def test_centile_risk_orders_scans_inversely_to_centile():
    risk = centile_risk([5, 50, 95])
    assert risk.tolist() == pytest.approx([0.95, 0.5, 0.05])
    y = np.array([1, 0, 0, 1, 0])
    centile = np.array([3.0, 60.0, 40.0, 8.0, 12.0])
    assert roc_auc_score(y, centile_risk(centile)) == pytest.approx(1 - roc_auc_score(y, centile))


def test_operating_point_uses_wilson_intervals():
    y = np.array([1] * 20 + [0] * 80)
    flagged = np.array([1] * 12 + [0] * 8 + [1] * 10 + [0] * 70)
    row = operating_point(y, flagged)
    assert (row["tp"], row["fn"], row["fp"], row["tn"]) == (12, 8, 10, 70)
    low, high = proportion_confint(12, 20, method="wilson")
    assert row["sensitivity"] == pytest.approx(0.6)
    assert (row["sensitivity_ci_low"], row["sensitivity_ci_high"]) == pytest.approx((low, high))
    assert row["ppv"] == pytest.approx(12 / 22)
    assert row["flagged_rate"] == pytest.approx(0.22)


def test_workload_matched_cut_off_flags_the_same_proportion_as_the_rule():
    rng = np.random.RandomState(5)
    prob = rng.uniform(size=2000)
    rule = rng.binomial(1, 0.17, 2000)
    threshold = workload_matched_threshold(prob, rule.mean())
    assert np.mean(prob >= threshold) == pytest.approx(rule.mean(), abs=1e-3)
    assert np.isnan(workload_matched_threshold(prob, 0.0))

    table = operating_point_table(rng.binomial(1, 0.1, 2000), prob, rule, (0.5, 0.1), "rule")
    assert table["strategy"].tolist() == ["rule", "model at 0.5", "model at 0.1", "model at matched workload"]
    assert table.iloc[-1]["flagged_rate"] == pytest.approx(rule.mean(), abs=1e-3)


def test_net_benefit_of_a_fixed_rule():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    flagged = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    # TP/n = 0.1, FP/n = 0.2, odds(0.2) = 0.25.
    assert net_benefit(y, flagged, 0.2) == pytest.approx(0.1 - 0.2 * 0.25)
    assert rule_net_benefit(y, flagged, [0.1, 0.2]) == pytest.approx(
        [0.1 - 0.2 / 9, 0.1 - 0.2 * 0.25]
    )
    treat_all = net_benefit(y, np.ones_like(y), 0.1)
    assert treat_all == pytest.approx(0.2 - 0.8 * 0.1 / 0.9)


def test_held_out_predictions_align_model_and_centile_rows(centile_reference, monkeypatch):
    from rebuttals.round3 import experiment_R3_5_efw_centile_comparator as r3_5

    monkeypatch.setattr(
        r3_5, "efw_centile", functools.partial(efw_centile, reference=centile_reference)
    )
    test_raw = pd.DataFrame({"ga": [140, 141, 140], "efw": [230.0, 380.0, 300.0]}, index=[7, 3, 9])
    external = SimpleNamespace(
        y_true=np.array([1, 0, 0]),
        country_arr=np.array([MALAYSIA, INDIA, MALAYSIA]),
        cluster_ids=np.array(["a", "b", "c"]),
        p_calibrated=np.array([0.4, 0.05, 0.1]),
        fold_data=SimpleNamespace(test_raw=test_raw),
    )
    frame = r3_5.held_out_predictions(external)
    assert len(frame) == 3
    assert frame["below_rule"].tolist() == [1, 0, 0]
    assert frame["efw"].tolist() == [230.0, 380.0, 300.0]
    assert frame["centile_risk"].iloc[0] > frame["centile_risk"].iloc[2]


# Tables S3 and S5


def test_smd_uses_root_mean_square_of_the_two_sds():
    assert standardized_mean_difference(10.0, 3.0, 7.0, 4.0) == pytest.approx(3.0 / np.sqrt(12.5))
    assert proportion_smd(0.5, 0.5) == 0.0
    assert proportion_smd(0.48, 0.50) == pytest.approx(-0.02 / np.sqrt((0.2496 + 0.25) / 2))


@pytest.mark.parametrize("delta", [0.0, 0.5, 1.0, 2.0])
def test_ovl_matches_closed_form_for_equal_variance_normals(delta):
    assert normal_overlap_coefficient(0.0, 1.0, delta, 1.0) == pytest.approx(
        2 * norm.cdf(-delta / 2), abs=1e-6
    )


def test_ovl_is_symmetric_and_shrinks_with_unequal_spread():
    a = normal_overlap_coefficient(0.0, 1.0, 0.0, 2.0)
    assert a == pytest.approx(normal_overlap_coefficient(0.0, 2.0, 0.0, 1.0))
    assert 0.0 < a < 1.0


def test_covariate_overlap_and_partition_summary():
    source = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "female": [1, 0, 1, 0]})
    target = pd.DataFrame({"x": [2.0, 3.0, 4.0, 5.0], "female": [1, 1, 0, np.nan]})
    table = covariate_overlap(source, target, ["x"], binary=["female"]).set_index("predictor")
    sd = source["x"].std()
    assert table.loc["x", "smd"] == pytest.approx(-1.0 / sd)
    assert table.loc["female", "target_mean"] == pytest.approx(2 / 3)

    target["sex"] = ["Female", "Female", "Male", None]
    source["sex"] = ["Female", "Male", "Female", "Male"]
    summary = partition_summary({"A": source, "B": target}, ["x"], categorical=["sex"])
    assert summary.loc[0, "A"] == "2.50±1.29"
    assert summary.set_index("predictor").loc["sex=Female", "B"] == "2 (66.67%)"


def test_cohort_shift_partitions_keep_test_pregnancies_out_of_training(prepared_cohort_dir, monkeypatch):
    from rebuttals.round3 import experiment_R3_6_cohort_shift_tables as r3_6
    from sga.pipeline.dataset import load_cohort

    monkeypatch.setattr(r3_6, "load_cohort", functools.partial(load_cohort, base_dir=prepared_cohort_dir))
    training, testing = r3_6.partitions("Malaysia")
    assert (testing["fold"] == EXTERNAL_TEST_FOLD).all()
    assert not set(training["id"]) & set(testing["id"])
    assert set(training["fetal_sex"].dropna()) <= {"Female", "Male"}
    assert (training["female"] == (training["gender"] == 0)).all()


# Figure 3, Table 3


def test_stratified_bootstrap_keeps_every_stratum_size():
    rng = np.random.RandomState(6)
    strata = [np.arange(0, 30), np.arange(30, 100)]
    for _ in range(20):
        idx = _stratified_indices(rng, strata)
        assert np.sum(idx < 30) == 30 and np.sum(idx >= 30) == 70


def test_outcome_stratified_bootstrap_never_loses_a_class():
    rng = np.random.RandomState(7)
    y = np.array([1] * 3 + [0] * 97)
    p = rng.uniform(size=100)
    point, low, high = bootstrap_metric_ci(y, p, strata=y, n_boot=300)
    assert low <= point <= high


def test_bootstrap_rejects_both_clusters_and_strata():
    with pytest.raises(ValueError):
        bootstrap_metric_ci([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], cluster_ids=[0, 0, 1, 1], strata=[0, 1, 0, 1])


def test_ensemble_predictions_average_the_fold_models(tmp_path):
    y = np.array([0, 1, 1])
    stack = [np.array([0.1, 0.6, 0.8]), np.array([0.3, 0.8, 0.6])]
    path = save_ensemble_predictions(tmp_path / "run", 4, y, stack, np.array([0, 0, 1]))
    assert path == ensemble_predictions_path(tmp_path / "run", 4)
    saved = pd.read_csv(path)
    assert saved["probability"].tolist() == pytest.approx([0.2, 0.7, 0.7])
    assert saved["country"].tolist() == [0, 0, 1]

    assert save_ensemble_predictions(tmp_path / "run", 4, y, [stack[0], stack[1][:2]], None) is None


# Table S10


def test_imputation_burden_labels():
    assert [imputation_burden(n) for n in (0, 1, 2, 3, 7)] == [
        "no common predictor imputed", "one predictor imputed", "two predictors imputed",
        "three or more imputed", "three or more imputed",
    ]


def test_stratum_with_too_few_events_is_not_estimable():
    frame = pd.DataFrame({LABEL: [1] * (MIN_EVENTS - 1) + [0] * 50, "prob": 0.1, "cluster": range(MIN_EVENTS + 49)})
    row = stratum_metrics(frame, "small")
    assert row["note"].startswith("not estimable") and "auroc" not in row


def _with_missing_add_on(pair, rng):
    complete, add_on = (frame.copy() for frame in pair)
    for column in ("efw", "hc", "ac"):
        add_on.loc[rng.uniform(size=len(add_on)) < 0.4, column] = np.nan
    return [complete, add_on]


@pytest.fixture(scope="module")
def scored_union(synthetic_cohorts):
    rng = np.random.RandomState(8)
    malaysia_pair, india_pair = (_with_missing_add_on(pair, rng) for pair in synthetic_cohorts)
    malaysia = build_union(malaysia_pair, MALAYSIA, True)
    india = build_union(india_pair, INDIA, False)
    return malaysia, india, score_all_records(malaysia, india, verbose=False)


def test_union_folds_are_grouped_by_pregnancy(scored_union):
    malaysia, india, _ = scored_union
    assert malaysia.groupby("id")["fold"].nunique().max() == 1
    assert set(malaysia["record_type"]) == {COMPLETE, INCOMPLETE}
    assert (malaysia.loc[malaysia["record_type"] == COMPLETE, "n_imputed_common"] == 0).all()
    assert india["cluster"].is_unique
    assert set(malaysia["n_imputed_common"]) - {0} and (malaysia["n_imputed_common"] <= len(COMMON_FEATURES)).all()


def test_every_record_scored_once_by_a_model_that_never_saw_it(scored_union):
    malaysia, india, predictions = scored_union
    assert len(predictions) == len(malaysia) + len(india)
    assert predictions["record_uid"].is_unique
    folds = pd.concat([malaysia, india]).set_index("record_uid")["fold"]
    assert (predictions.set_index("record_uid")["fold"] == folds.loc[predictions["record_uid"]]).all()
    assert predictions["prob"].between(0, 1).all()


def test_record_type_and_burden_tables_partition_the_records(scored_union):
    *_, predictions = scored_union
    by_type = table_by_record_type(predictions, n_boot=50).set_index("stratum")
    total = by_type.loc["All records", "n_scans"]
    assert by_type.loc["Complete records", "n_scans"] + by_type.loc["Incomplete records", "n_scans"] == total
    cohort_rows = [r for r in by_type.index if "(" in r]
    assert by_type.loc[cohort_rows, "n_scans"].sum() == total

    by_burden = table_by_imputation_burden(predictions, n_boot=50)
    assert by_burden["n_scans"].iloc[1:].sum() == total

    delong = delong_complete_vs_incomplete(predictions).set_index("cohort")
    assert delong.loc["Total", "n_complete"] + delong.loc["Total", "n_incomplete"] == total
