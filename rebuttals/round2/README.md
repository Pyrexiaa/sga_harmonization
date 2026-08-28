# Supplementary analyses, round 2

Second set of supplementary analyses for *A unified feature-harmonization
framework for SGA prediction* (Malaysia n = 18,582; India n = 1,760).

Every script imports its pipeline and evaluation helpers from the `sga` package —
none of them re-implements analysis code. Results are written under
`sga.config.ROUND2_DIR` (`Results/rebuttal_round2/<experiment>/`).

Run from the repository root:

```
python -m rebuttals.round2.experiment_R2_1_size_matched_country
```

`docs/MANUSCRIPT_MAP.md` maps every table and figure to the script that
produces it.

## One external fold, four analyses

`R2_3`, `R2_3b`, `R2_4` and `R2_5` all report the same external test fold. They build
it through a single shared constructor,
`sga.pipeline.external_fold.build_calibrated_external_fold`, which:

1. fits four fold-models on the harmonized feature set, one per development fold,
   each on the other three — with its own within-feature imputer, cross-domain
   imputers and scaler, all fitted on those training rows;
2. scores each fold-model on its own held-out fold, giving both a validation AUROC
   and one authentic out-of-fold prediction per development record;
3. fits ONE Platt calibrator on those pooled held-out predictions and applies it
   unchanged to fold 4 — the fold-4 labels never enter it;
4. selects the operating points on the development block through a *cross-fitted*
   calibrator, so no record is scored by a map its own label helped fit;
5. carries the fold-model with the **highest validation AUROC** onto fold 4
   (Methods 2.3.4), with that fold's imputers and scaler — no model is refitted on
   the pooled development block;
6. returns the cohort labels and the pregnancy cluster keys alongside the
   probabilities.

Step 2 matters more than it looks. SMOTENC touches training rows only, so a fold's
held-out rows sit at the cohort's natural prevalence. Taking the out-of-fold
predictions over the assembled *training* matrix instead would anchor the Platt map
to a 50% base rate and apply it to a fold at ~21.6%, where it cannot reduce the
calibration error and no threshold chosen on it transfers. The builder prints both
prevalences on every run.

Because the four scripts share step 1–5, the 0.50 row of the Table S4 sweep is
guaranteed to reproduce the sensitivity quoted in the Results, and the fairness
section is guaranteed to describe the same 853 scans as the calibration table.
When each script built its own copy, that was not guaranteed and had in fact
drifted.

## Which confidence interval belongs to which number

The Statistical Analysis section splits these two ways, and so does the code:

| Quantity | Method | Implementation |
|---|---|---|
| Sensitivity, specificity, PPV, NPV, cohort TPR / FPR — any proportion | Wilson score | `sga/evaluation/proportions.py::wilson_ci` |
| Any difference between two independent proportions | Newcombe hybrid score | `sga/evaluation/proportions.py::newcombe_difference_ci` |
| AUROC, AUPRC, ECE, Brier | Percentile bootstrap, 2,000 iterations, resampling whole pregnancies | `sga/evaluation/bootstrap.py` |

The bootstrap is reported alongside the score intervals for the fairness gaps and
the swept rates, so the two can be compared — but the score intervals are the
reported values. With 15 Indian SGA events a percentile bootstrap of an absolute
difference is biased away from zero and can return an interval that cannot cover 0.

| Script | Manuscript element / claim it produces | Outputs (under `Results/rebuttal_round2/`) |
|---|---|---|
| `experiment_R2_1_size_matched_country.py` | The size-matched sensitivity analysis in §3.2: held-out **India** AUROC/AUPRC for India-only, Malaysia-only, pooled-common and pooled-harmonized training at a **matched** training size, with bootstrap 95% CIs over 4 folds × 10 subsample seeds, plus per-fold paired DeLong for harmonized vs common. | `R2_1_size_matched_country/india_per_fold_repeat.csv`, `india_summary_ci.csv`, `delong_harmonized_vs_common_india.csv` |
| `experiment_R2_3_calibration_uncertainty.py` | **Table 3** — ECE and Brier with bootstrap 95% CIs (total / Malaysia / India) — and §3.2's operating points: the default 0.50, the 0.10 screening cut-off, the validation-selected Youden point (0.46) and a sensitivity ≥ 0.80 threshold, each with Wilson intervals on every rate. | `R2_3_calibration_uncertainty/table3_calibration_by_cohort.csv`, `calibration_uncertainty_by_country.csv`, `test_fold_composition.csv` |
| `experiment_R2_3b_threshold_sweep.py` | **Appendix Table S4** — sensitivity and specificity with Wilson 95% CIs at every cut-off from 0.10 to 0.90, per cohort, plus PPV/NPV at the observed prevalence, decision-curve net benefit and the bootstrap bounds for comparison. | `R2_3b_threshold_sweep/tableS4_sensitivity.csv`, `tableS4_specificity.csv`, `threshold_sweep_by_country.csv`, `threshold_sweep.png` |
| `experiment_R2_4_cluster_inference.py` | §3.8 "Pregnancy-Cluster-Aware Analysis": unique-pregnancy counts and the scans-per-pregnancy distribution, a **pregnancy-cluster** bootstrap 95% CI for AUROC and AUPRC (2,000 iterations, unit = pregnancy, seed 123), and a one-index-scan-per-pregnancy sensitivity analysis. | `R2_4_cluster_inference/pregnancy_descriptives.csv`, `malaysia_scans_per_pregnancy.csv`, `cluster_bootstrap_auroc.csv`, `index_scan_sensitivity.csv` |
| `experiment_R2_5_fairness_uncertainty.py` | **The whole of §3.9 and appendix Table S5.** Signed Malaysia-minus-India gaps on the CALIBRATED probabilities across the 0.10–0.90 grid with Newcombe intervals, per-cohort rates with Wilson intervals, the 807/169 and 46/15 strata counts, per-cohort AUROC/AUPRC/ECE/Brier with the cluster bootstrap, and a formatted block of the numbers the section quotes. | `R2_5_fairness_uncertainty/tableS5_cohort_fairness.csv`, `fairness_calibrated_threshold_sweep.csv`, `fairness_at_default_0.50.csv`, `fairness_at_screening_0.10.csv`, `discrimination_by_cohort_ci.csv`, `fairness_bootstrap_ci.csv`, `fairness_manuscript_numbers.txt`, `fairness_vs_threshold.png` |
| `experiment_R3_2_india_auroc_delong.py` | Whether the India AUROC gain (country-specific baseline **0.5605** → unified **0.6753**) is statistically distinguishable: per-fold AUROC difference, its bootstrap 95% CI, whether that CI excludes zero, the paired DeLong p per fold and its median, and the total India SGA event count. Backs the Conclusion's "did not reach statistical significance". | `R3_2_india_auroc_delong/india_delong_per_fold.csv`, `india_delong_summary.csv` |
| `experiment_R3_5_imputed_vs_native.py` | How much of the AUPRC gain from `ute_ari`/`af` is imputation-driven: AUPRC (common vs harmonized) with bootstrap 95% CIs pooled across folds **per cohort** — the Malaysia delta is the natively measured contribution, the India delta the imputed one (+0.016 in §4.1). | `R3_5_imputed_vs_native/imputed_vs_native_auprc.csv` |

## Shared settings

All scripts use `SEED = 123` (`sga.config.SEED`), folds 0–3 for cross-validation
(`N_FOLDS_CV`) with fold 4 as the external test partition (`EXTERNAL_TEST_FOLD`),
2,000 bootstrap iterations (`N_BOOTSTRAP`), the 0.5 default decision threshold
(`DECISION_THRESHOLD`), the 0.10 screening threshold (`SCREENING_THRESHOLD`), the
0.10–0.90 reporting grid (`THRESHOLD_GRID`) and the harmonized feature set
`["ute_ari", "af"]` (`HARMONIZED_SELECTED_FEATURES`).

Note that `R1.experiment_R2_8_fairness_metrics` also prints disparity numbers. Those
are uncalibrated, absolute and interval-free, computed across the development folds;
they are a supporting stability check and **will not match** §3.9.
