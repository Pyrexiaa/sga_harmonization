# Supplementary analyses, round 1

First set of supplementary analyses for *A unified feature-harmonization
framework for SGA prediction* (Malaysia n = 18,582; India n = 1,760).

Every script imports its pipeline, model-loading and evaluation helpers from the
`sga` package (`sga.pipeline.harmonized_fold.prepare_fold`,
`sga.pipeline.train_unified`, `sga.pipeline.model_io`, `sga.evaluation.*`,
`sga.models.domain_adaptation`) — none of them re-implements analysis code.
Results are written under `sga.config.ROUND1_DIR`
(`Results/rebuttal_round1/<experiment>/`).

Run from the repository root:

```
python -m rebuttals.round1.experiment_R0_baseline_retrain
```

`docs/MANUSCRIPT_MAP.md` maps every table and figure to the script that
produces it.

## Order of execution

`experiment_R0_baseline_retrain.py` **must run first** — it regenerates the
imputers and the unified/single-source weights that every other script loads. Its
`__main__` runs inference only; the `retrain_*` phases are invoked explicitly
(see the comment block in `run_all.sh`). `experiment_R0_baseline_retrain_manual.py`
then writes the harmonized (`ute_ari` + `af`) weights that R2-3, R2-8, R2-9 and
R2-10 consume.

## Scripts

| Script | Manuscript element | Prerequisites | Output directory (under `Results/rebuttal_round1/`) |
|---|---|---|---|
| `experiment_R0_baseline_retrain.py` | Weights behind every round-1 number; the cross-domain arm of Figure 3 comes from `retrain_single_source_common_all()` | none — run first | `R0_baseline_retrain/` |
| `experiment_R0_baseline_retrain_manual.py` | Harmonized `ute_ari`+`af` weights used by Tables 2, 3, 4, 6, 7 and Figures 4 and 6 | R0 | `R0_baseline_retrain/selected_imputation/<combo>/` |
| `experiment_R1_1_remove_prev_pregnancy.py` | Maternal-history ablation behind the "before" panel of Figure 6 (weights only, no metric CSV) | R0 | `R1_1_remove_prev_pregnancy/` |
| `experiment_R1_2_data_scaling.py` | Figure 7 — training half | R0 | `R1_2_data_scaling/` |
| `experiment_R1_2_data_scaling_inference.py` | Figure 7 — external-fold scoring of the size sweep; plotted by `scripts/05c` | R1-2 training | `R1_2_data_scaling/` (`per_country_eval_all_*.csv`) |
| `experiment_R2_1_domain_generalization_comparison.py` | **Table 5** (CORAL / IRM / DANN vs the unified framework) | R0 | `R2_1_domain_generalization/` |
| `experiment_R2_1b_comparator_settings.py` | **Appendix Table S6** — dumps the comparator settings from the R2-1 constants; trains nothing | R2-1 present on disk (import only) | `R2_1b_comparator_settings/` |
| `experiment_R2_2_imputation_metrics.py` | **Appendix Tables S2a and S2b** (per-feature imputation quality) | none — this is the imputer-training step | `R2_2_imputation_metrics/` |
| `experiment_R2_3_additional_metrics.py` | **Table 2** (ECE/Brier before vs after Platt), **Table 4** (calibrated AUROC/AUPRC/threshold metrics) and **Figure 4** (DCA, reliability diagrams). Table 3's CIs come from round 2 | R0, R0-manual | `R2_3_additional_metrics_and_dca/` |
| `experiment_R2_4_decision_curve_analysis.py` | Standalone decision curve for the unified CatBoost (Figure 4 is produced by R2-3) | R0 | `R2_4_decision_curve_analysis/` |
| `experiment_R2_5_imputation_ablation.py` | **Table 7** (no-imputation / mean / mode / iterative strategies) | R0 | `R2_5_imputation_ablation/` |
| `experiment_R2_6_weighted_training.py` | **Table 8** (equal-country and sqrt-ratio domain weighting) | R0 | `R2_6_weighted_training/` |
| `experiment_R2_7_split_unified_by_country.py` | Unified results reported total / Malaysia / India | R0 | `R2_7_split_unified_by_country/` |
| `experiment_R2_8_fairness_metrics.py` | **Supporting only.** Uncalibrated, absolute, interval-free disparity of the pretrained fold-models across the development folds — a stability check on the *direction* of the gap. The manuscript's §3.9 numbers and appendix Table S5 come from `R2.experiment_R2_5_fairness_uncertainty` (calibrated probabilities, signed gaps, Wilson/Newcombe intervals, external fold) and will not match these | R0, R0-manual, R2-5 | `R2_8_fairness_metrics/` |
| `experiment_R2_9_delong_test.py` | **Table 6** (`p_value_holm_bonferroni`, Holm-Bonferroni corrected; the uncorrected `p_value` column is retained but not tabulated in the paper), pairwise across the six classifiers | R0, R0-manual | `R2_9_delong_test/` |
| `experiment_R2_10_internal_vs_external.py` | **Appendix Table S3** and the "Model Internal Validation Performances" optimism figures | R0, R0-manual | `R2_10_internal_vs_external/` |
| `evaluate_per_country.py` | Per-cohort metrics for every saved model, with and without maternal history | R0, R1-1 | `per_country_no_retrain/` |
| `shap_without_retraining.py` | **Figure 6** (per-cohort SHAP feature rankings) | R0, R0-manual, R1-1 | `per_country_shap_no_retrain/` |

## Shared settings

All scripts use `SEED = 123` (`sga.config.SEED`), folds 0-3 for cross-validation
(`N_FOLDS_CV`) with fold 4 as the held-out external partition
(`EXTERNAL_TEST_FOLD`), 2000 bootstrap iterations (`N_BOOTSTRAP`), a 0.5 default
decision threshold (`DECISION_THRESHOLD`), 10 calibration bins (`ECE_BINS`) and
the harmonized feature set `["ute_ari", "af"]`
(`HARMONIZED_SELECTED_FEATURES`).

One inconsistency to be aware of: the "Neural Network" row is DNN config 0 in
`experiment_R2_3_additional_metrics` and `experiment_R2_10_internal_vs_external`,
but config 2 in `experiment_R2_9_delong_test`. Fix the configuration before
quoting a Neural Network number across tables.
