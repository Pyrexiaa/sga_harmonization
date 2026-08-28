# Manuscript to code map

Maps every numbered table and figure of the paper to the entry point that produces it,
the file that entry point writes, and the column to read.

Run all commands from the repository root. Throughout: `R1` = `rebuttals/round1`,
`R2` = `rebuttals/round2`; results land under `Results/rebuttal_round1/` and
`Results/rebuttal_round2/` respectively (`sga.config.ROUND1_DIR` / `ROUND2_DIR`).

---

## Methods

| Manuscript element | Produced by |
|---|---|
| Study population, screening, eligibility (Figure 1 counts) | `scripts/01a_prepare_malaysia.py`, `scripts/01b_prepare_india.py` — each prints a CONSORT-style flow and writes `consort_flow_i21.csv` (`step`, `n`) |
| SGA label (birthweight ≤ 10th INTERGROWTH-21st centile) | `sga/data/centiles.py::merge_groundtruth` |
| Patient-grouped stratified 5-fold split; add-on records restricted to training | `sga/data/splits.py::split_complete_and_addon` |
| Leakage-safe cross-validation and calibration pipeline | `sga/pipeline/harmonized_fold.py::prepare_fold`, assembled for the external fold by `sga/pipeline/external_fold.py::build_calibrated_external_fold`. Step by step in `docs/PIPELINE.md` |
| Within-feature iterative imputation fitted on the training partition | `sga/pipeline/dataset.py::impute_within_feature` — fitted inside each fold on its training rows, applied unchanged to the held-out rows |
| "the best model, judging from the highest AUROC based on the validation sets, was used to evaluate on the testing data" | `sga/pipeline/external_fold.py::development_fold_models` scores each fold-model on its own held-out fold; the best is carried onto fold 4 with its own imputers and scaler |
| Platt scaling fitted by cross-fitting on held-out validation predictions | `sga/evaluation/calibration.py::fit_platt` + `apply_platt`, fitted on the folds 0–3 out-of-fold predictions and applied unchanged to fold 4. Asserted by `tests/test_manuscript_alignment.py` |
| Ten common features | `sga.config.COMMON_FEATURES` |
| Feature-specific cross-domain imputation (the harmonization stage) | `scripts/02_train_imputers.py`, `sga/imputation/apply.py` |
| Imputation-quality retention criteria (binary AUROC > 0.80 **and** F1 > 0.50; multiclass F1 > 0.80 **and** AUROC > 0.50; continuous R² > 0.80) | `sga/imputation/apply.py::select_features_with_threshold`; thresholds in `sga.config` |
| Retained cross-domain features (`ute_ari`, `af`) | `sga.config.HARMONIZED_SELECTED_FEATURES` |
| Statistical analysis — t-test, chi-square | `sga/data/statistics.py` |
| Statistical analysis — **Wilson score** intervals for proportions and **Newcombe hybrid score** intervals for their differences | `sga/evaluation/proportions.py` |
| Statistical analysis — bootstrap CIs (2,000 iterations) for AUROC, AUPRC, ECE, Brier | `sga/evaluation/bootstrap.py` |
| Statistical analysis — DeLong's test | `sga/evaluation/delong.py` |

**Which interval method applies where** is not a detail: the manuscript specifies score
intervals for every proportion and the bootstrap only for the four metrics that are not
proportions. `sga.config.CI_METHODS` records the mapping and
`tests/test_manuscript_alignment.py` enforces it.

---

## Main manuscript

| Element | Command | Output file → column |
|---|---|---|
| **Table 1** — patient characteristics, AGA vs SGA per cohort | `python scripts/01c_cohort_characteristics.py --cohort both` | `Results/tables/table1_{malaysia,india}_{continuous,categorical}_i21.csv` — one file per cohort × variable type; the side-by-side layout is assembled by hand. `P-value` is Student's t-test (as the caption states); `Welch P-value` and `Levene P-value` are reported alongside as a sensitivity check |
| **Table 2** — ECE and Brier before vs after Platt scaling, six models | `python -m R1.experiment_R2_3_additional_metrics` | `R2_3_additional_metrics_and_dca/calibration/calibration_metrics_by_country.csv` → rows with `eval_split=total`, both values of `calibration` (`before`/`after`); columns `ece`, `brier_score` |
| **Table 3** — ECE and Brier with 95% CIs, Logistic Regression, Total / Malaysia / India | `python -m R2.experiment_R2_3_calibration_uncertainty` | `R2_3_calibration_uncertainty/table3_calibration_by_cohort.csv`; long form with bounds in `calibration_uncertainty_by_country.csv` → `ece`, `ece_ci_low/high`, `brier`, `brier_ci_low/high` |
| **Table 4** — calibrated AUROC, AUPRC, sensitivity, specificity, PPV, NPV | `python -m R1.experiment_R2_3_additional_metrics` | same `calibration_metrics_by_country.csv` → `calibration=after`, `eval_split=total`; columns `roc_auc`, `auprc`, `sensitivity`, `specificity`, `ppv`, `npv` |
| **Table 5** — CORAL / IRM / DANN vs the unified framework | `python -m R1.experiment_R2_1_domain_generalization_comparison` | `R2_1_domain_generalization/summary_comparison_by_country.csv` → `method` × `eval_split` (`total`/`malaysia`/`india`), columns `roc_auc_mean`, `auprc_mean`. The `Naive_*` rows are extra context, not part of Table 5 |
| **Table 6** — pairwise DeLong p-values, total cohort, **Holm–Bonferroni** | `python -m R1.experiment_R2_9_delong_test` | `R2_9_delong_test/delong_ensemble_fold4.csv` → `eval_split=total`, column **`p_value_holm_bonferroni`** (the family is the 15 pairs within one `eval_split`). The uncorrected values remain in `p_value` |
| **Table 7** — ablation: (a) cross-domain imputation, (b) within-feature imputation method | `python -m R1.experiment_R2_5_imputation_ablation` | `R2_5_imputation_ablation/summary_comparison_by_country.csv` → `strategy` ∈ `A_no_imputation` (intersection only), `B_mean`/`C_mode` (mean/mode), `E_iterative` (ours) |
| **Table 8** — ablation: domain weighting | `python -m R1.experiment_R2_6_weighted_training` | `R2_6_weighted_training/summary_comparison.csv` → `strategy` ∈ `A_equal_country_weight` (domain-balanced), `C_no_weight_baseline` (unweighted, ours) |
| **Figure 1** — data-preparation flowchart | counts from `scripts/01a`/`01b` | the diagram itself is drawn externally |
| **Figure 2** — framework schematic | — | conceptual diagram, no code |
| **Figure 3** — AUROC of six classifiers, three arms per cohort | `python scripts/05a_figure3_auroc_comparison.py` | `Results/figures/figure3/figure3_auroc_values.csv` → `cohort`, `model`, `strategy`, `auroc`, `ci_low`, `ci_high`. Needs stages 3–4 for the baseline (orange) and unified (blue) arms, **and**, for the cross-domain (green) arm, `retrain_single_source_common_all()` followed by `python -m R1.experiment_R0_baseline_retrain`, which scores those weights via `run_single_source_common_inference()`. 05a aborts rather than quietly drawing a two-armed figure |
| **Figure 4** — decision curves, full range and 5–20% range | `python -m R1.experiment_R2_3_additional_metrics` | `R2_3_additional_metrics_and_dca/dca_ALL_models_after.png` + `dca_data_all_models.csv`. Standalone single-model version: `R1.experiment_R2_4_decision_curve_analysis` |
| **Figure 5** — subgroup AUROC by gestational week and maternal age | `python scripts/05b0_subgroup_inference.py` once per arm, then `python scripts/05b_figure5_subgroup_analysis.py` | `Results/figures/figure5/figure5_auroc_{gestational_week,maternal_age}_{pooled,malaysia,india}.png` and `figure5_values.csv`. All three arms (`unified`, `malaysia_baseline`, `india_baseline`) are required — exact flags in `README.md` §4 |
| **Figure 6** — SHAP importance, before and after removing unreliably imputed features | `python -m R1.shap_without_retraining` | `per_country_shap_no_retrain/shap_meanabs_<model>_{malaysia,india}.csv` → `name`, `mean_abs_shap` |
| **Figure 7** — AUROC vs training-set size, and % difference vs the 1,000-sample model | `python -m R1.experiment_R1_2_data_scaling`, then `R1.experiment_R1_2_data_scaling_inference`, then `python scripts/05c_figure7_sample_size.py` | `Results/figures/figure7/figure7_combined.png/.pdf` (both panels, shared legend), `figure7_auroc.*`, `figure7_pctdiff.*`, `figure7_by_cohort.*`, and `figure7_plotted_values.csv` → `cohort`, `model`, `training_size`, `auroc`, `pct_diff_vs_1000` |

### Narrative results (numbers quoted in the text, no table of their own)

| Claim | Command | Output → column |
|---|---|---|
| §3.2 — validation-selected Youden operating point 0.46 (sensitivity 0.32, specificity 0.93) | `python -m R2.experiment_R2_3_calibration_uncertainty` | `calibration_uncertainty_by_country.csv` → `youden_threshold`, `sensitivity@youden`, `specificity@youden` (each with a `_ci` column) |
| §3.2 — size-matched sensitivity analysis: India-only 0.573, Malaysia-only 0.658, pooled-common 0.653, pooled-harmonized 0.653 | `python -m R2.experiment_R2_1_size_matched_country` | `R2_1_size_matched_country/india_summary_ci.csv` → `arm`, `metric=auroc`, `mean`, `ci_low`, `ci_high` |
| §3.3 — LR internal 0.803 vs test 0.811; ≤ 0.017 gap for five models; CatBoost ≈ 0.055 | `python -m R1.experiment_R2_10_internal_vs_external` | `R2_10_internal_vs_external/optimism_summary.csv` → `mean_validation_auroc`, `mean_test_auroc`, `validation_minus_test` |
| §3.8 — 10,914 pregnancies, mean 1.70 scans, max 8; cluster-bootstrap AUROC 0.812 / 0.823 / 0.684; index-scan 0.820 / 0.834 / 0.684 | `python -m R2.experiment_R2_4_cluster_inference` | `R2_4_cluster_inference/pregnancy_descriptives.csv`, `cluster_bootstrap_auroc.csv` (`split`, `auroc`, `auroc_cluster_ci`, `n_pregnancies`), `index_scan_sensitivity.csv` |
| §3.9 — the whole Cohort-Level Fairness section: the 807/169 and 46/15 strata counts, the rates and gaps at the 0.50 and 0.10 cut-offs, and their Wilson / Newcombe intervals | `python -m R2.experiment_R2_5_fairness_uncertainty` | `R2_5_fairness_uncertainty/fairness_manuscript_numbers.txt` (formatted), `test_fold_composition.csv`, `fairness_at_default_0.50.csv`, `fairness_at_screening_0.10.csv`, `discrimination_by_cohort_ci.csv` |
| §4.1 — the AUPRC gain is India-side (+0.016) and ≈ 0 in Malaysia | `python -m R2.experiment_R3_5_imputed_vs_native` | `R3_5_imputed_vs_native/imputed_vs_native_auprc.csv` → `cohort`, `auprc_common`, `auprc_harmonized`, `auprc_gain` |
| §5 / Conclusion — is the India gain distinguishable from zero? | `python -m R2.experiment_R3_2_india_auroc_delong` | `R3_2_india_auroc_delong/india_delong_summary.csv` → `diff_ci_low/high`, `diff_ci_excludes_zero`, `median_delong_p` |

---

## Appendix

| Element | Command | Output → column |
|---|---|---|
| **Table S1** — feature availability per cohort | `python scripts/01c_cohort_characteristics.py` | `Results/tables/tableS1_feature_availability_i21.csv` → `cohort`, `feature`, `measured`, `n_observed`, `pct_observed` |
| **Table S2a** — categorical imputation quality (AUROC, F1) | `python -m R1.experiment_R2_2_imputation_metrics` | `R2_2_imputation_metrics/classification_imputation_metrics.csv` → binary: `test_ROC_AUC`, `test_F1`; multiclass: `test_ROC_AUC_ovr`, `test_F1_weighted`. The `Retained` column is the gate in `sga/imputation/apply.py` applied to these values |
| **Table S2b** — continuous imputation quality (MAE, RMSE, R²) | same run | `R2_2_imputation_metrics/regression_imputation_metrics.csv` → `test_MAE`, `test_RMSE`, `test_R2` |
| **Table S3** — per-fold internal cross-validation vs external test, six models | `python -m R1.experiment_R2_10_internal_vs_external` | `R2_10_internal_vs_external/tableS3_internal_vs_external.csv` → `Model`, `Metric`, `Set` (`Validation`/`Test`), `Fold 0`–`Fold 3`, `Mean` |
| **Table S4** — threshold sweep 0.10–0.90, sensitivity and specificity with 95% CIs | `python -m R2.experiment_R2_3b_threshold_sweep` | `R2_3b_threshold_sweep/tableS4_sensitivity.csv`, `tableS4_specificity.csv` (thresholds × cohorts, Wilson intervals); long form with bounds, PPV/NPV at observed prevalence, net benefit and the bootstrap comparison in `threshold_sweep_by_country.csv` |
| **Table S5** — cohort-level fairness across calibrated-probability thresholds | `python -m R2.experiment_R2_5_fairness_uncertainty` | `R2_5_fairness_uncertainty/tableS5_cohort_fairness.csv` (Newcombe intervals, with the bootstrap interval in a parallel column); long form in `fairness_calibrated_threshold_sweep.csv` |
| **Table S6** — implementation details of the domain-generalization comparators | `python -m R1.experiment_R2_1b_comparator_settings` | `R2_1b_comparator_settings/tableS6_comparator_settings.csv` — read straight from the constants in `R1.experiment_R2_1_domain_generalization_comparison`, so the appendix cannot drift from the code |

---

## Supporting analyses (no numbered element of their own)

| Script | Purpose |
|---|---|
| `R1.experiment_R0_baseline_retrain` | Regenerates the imputers and all unified / single-source weights. **Prerequisite for every other round-1 script.** Its `__main__` runs inference only; the `retrain_*` phases must be called explicitly (see `run_all.sh`) |
| `R1.experiment_R0_baseline_retrain_manual` | Trains the manually selected imputation-feature combinations, including the harmonized `sel_ute_ari_af` set that Tables 2, 3, 4, 6, 7 and Figures 4 and 6 read |
| `R1.experiment_R1_1_remove_prev_pregnancy` | Removes the India-only maternal-history features that imputed at or below chance (the "before" arm of Figure 6) |
| `R1.experiment_R2_7_split_unified_by_country` | Splits the unified model's predictions by cohort |
| `R1.experiment_R2_8_fairness_metrics` | **Supporting only.** Uncalibrated, absolute, interval-free disparity of the pretrained fold-models across the development folds. The manuscript's §3.9 numbers come from `R2.experiment_R2_5_fairness_uncertainty` and will not match these |
| `R1.evaluate_per_country` | Per-cohort metrics for every saved model, with and without maternal history |
