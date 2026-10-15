# Supplementary analyses, round 3

Third set of supplementary analyses. Every script imports its pipeline and evaluation
helpers from the `sga` package and writes under `sga.config.ROUND3_DIR`
(`Results/rebuttal_round3/<experiment>/`). Run from the repository root, after the
imputers (`scripts/02_train_imputers.py`) have been trained.

| Script | Produces | What it does |
|---|---|---|
| `experiment_R2_2_af_diagnostic` | Table S2c | Class distribution of amniotic fluid condition, the af imputer's validation-fold confusion matrix, weighted F1 and macro one-vs-rest AUROC (95% CI) against an always-predict-majority baseline, and the af levels it writes into the Indian cohort |
| `experiment_R2_2b_af_downstream` | Table 7a decision | Common features, + ute_ari, and + ute_ari + af through the calibrated external-fold construction, compared on identical rows by paired DeLong tests and paired bootstrap differences |
| `experiment_R3_1_incomplete_records` | Table S10 | Complete and incomplete records pooled, folds reassigned by pregnancy, every record scored out of fold; strata by record type and by the number of common predictors imputed; unpaired DeLong for complete vs incomplete |
| `experiment_R3_5_efw_centile_comparator` | Table S9, Section 3.5, Figure 4 | EFW centile alone vs the model on the same external fold: AUROC/AUPRC (pregnancy-cluster bootstrap), operating points (Wilson) for the EFW < 10th centile rule, the model at 0.50 and 0.10 and at matched workload, paired DeLong, and net benefit |
| `experiment_R3_6_cohort_shift_tables` | Tables S3 and S5 | Training vs testing partition summaries, and SMD and OVL between the Malaysian (source) and Indian (target) cohorts |

```
python -m rebuttals.round3.experiment_R2_2_af_diagnostic
python -m rebuttals.round3.experiment_R2_2b_af_downstream
python -m rebuttals.round3.experiment_R3_1_incomplete_records
python -m rebuttals.round3.experiment_R3_5_efw_centile_comparator
python -m rebuttals.round3.experiment_R3_6_cohort_shift_tables
```

## Notes on method

* **Multiclass imputation gate.** A weighted F1 is dominated by the majority level, so
  amniotic fluid condition is also required to beat an always-predict-majority
  baseline and to have a macro one-vs-rest AUROC whose 95% CI excludes 0.50. It fails
  both, so the harmonized feature set is the ten common predictors plus `ute_ari`
  (`sga.config.HARMONIZED_SELECTED_FEATURES`).
* **EFW centile.** `sga.data.centiles.efw_centile` interpolates inside the tabulated
  2.5th-97.5th range and extrapolates beyond it, so no held-out scan is dropped. The
  rule flags a scan when its centile is below 10.
* **Incomplete records.** Within-feature imputation, the cross-domain imputers,
  SMOTENC, standardisation, the grid search and the Platt calibrator are all fitted on
  the four development folds and applied unchanged to the held-out fold. A stratum
  with fewer than five SGA events is reported as not estimable. The complete and
  incomplete strata are disjoint, so they are compared by the unpaired DeLong test.
