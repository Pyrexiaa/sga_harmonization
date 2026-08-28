# The leakage-safe cross-validation and calibration pipeline

Every data-dependent operation is fitted on training data only and then applied unchanged to
held-out data. This document gives the exact fold-by-fold sequence, mirroring the Methods
subsection "Leakage-Safe Cross-Validation and Calibration Pipeline".

## Partitioning

Each cohort is partitioned by **patient** into five folds using stratified group k-fold
assignment (`sga/data/splits.py::assign_folds`), so all third-trimester scans from one
pregnancy fall in the same fold. The India cohort has one record per pregnancy, so plain
stratified k-fold is equivalent there.

* **Folds 0-3** — model development (four-fold cross-validation).
* **Fold 4** — external test partition. It is never seen during imputer fitting, feature
  selection, resampling, hyperparameter tuning, calibration or threshold selection.

Records with remaining missing values form the **add-on** set (`fold = -1`). They only ever
enter a training partition, and pregnancy `id` is matched at two levels so that no add-on scan
of a held-out pregnancy can reach training:

* `sga/pipeline/dataset.py::load_cohort` removes add-on scans belonging to **fold-4**
  pregnancies at load time, alongside the fold-4 complete-case rows. This has to happen at
  load time: the add-on rows carry `fold = -1`, so a fold-label filter alone never touches
  them.
* `sga/pipeline/dataset.py::process_raw_train_and_test_df` then removes add-on scans belonging
  to the **current CV fold's** test pregnancies.
* `sga/data/splits.py::split_complete_and_addon` writes the add-on records with their gaps
  intact (`impute_data=False`). Nothing is imputed at preparation time, so no imputer is ever
  fitted across folds, and the written tables still carry every record the cohort counts
  describe. The fold-4 exclusion happens at load time, above, which is the only place that
  knows whether a development or an external run is being set up.

## Per-fold sequence

Implemented end to end in `sga/pipeline/harmonized_fold.py::prepare_fold`.

1. **Split.** Build the train/test partition for the fold; merge the add-on records into the
   training side only.
2. **Cross-domain imputation.** The feature-specific imputation models are **refitted on this
   fold's own training partition** (`sga/imputation/fold_imputers.py::fit_fold_imputers`) from
   the ten shared input features, and handed to `impute_df` in memory. A single imputer fitted
   once on the whole development block and reused for every fold would let a fold's held-out
   rows shape the model that generates their own harmonized features, so the refit is per fold.
   (`fit_imputers_per_fold=False` loads the persisted imputers instead; it exists for
   diagnostics and is not the reported configuration.)

   The refits reuse the hyperparameters tuned by `scripts/02_train_imputers.py`, so a per-fold
   model differs from the persisted one only in the rows it saw. Class imbalance is handled
   with balanced class weights rather than SMOTENC, so the imputed values do not depend on a
   resampling RNG path.

   Which candidate features are **retained** is a development-block decision, taken once from
   the persisted out-of-fold metrics (`select_features_with_threshold`) and never informed by
   fold 4. The criteria are per type, not one shared scalar:
   * Binary: AUROC > 0.80 primary, F1 > 0.50 secondary safeguard.
   * Multiclass: F1 > 0.80 primary, AUROC > 0.50 secondary check.
   * Continuous: R² > 0.80, with MAE inspected as a complementary measure.

   Of all candidates, only amniotic fluid condition (`af`) and uterine artery resistance
   index (`ute_ari`) satisfy these criteria. The maternal-history features imputed at or
   below chance and are excluded.
3. **Within-feature imputation.** Remaining gaps are completed by an iterative
   (MICE-style) imputer fitted on **this fold's training rows only**
   (`sga/pipeline/dataset.py::impute_within_feature`, built on
   `sga/data/cleaning.py::fit_iterative_imputer` / `apply_iterative_imputer`).

   With four development folds, the imputer for fold 3 is fitted on folds 0-2 and then
   applied unchanged to fold 3. Fitting once on the pooled development block would let the
   rows held out as fold 3 shape the model that fills fold 3's own training data. Add-on
   records only ever enter a training partition, so no test row's value is produced by this
   step, and the training data is independent of the fold being scored.
   `tests/test_leakage.py` checks this behaviourally: replacing the held-out fold with
   absurd values must leave the training imputations bit-identical.

   The outcome column is excluded from the imputation model, since it is available for
   training rows but not for the rows a deployed model would score. Modelling
   inter-feature relationships reconstructs values more faithfully than marginal
   mean/mode substitution (manuscript Table 7b).
4. **Feature-space restriction.** Keep the common features plus the retained cross-domain
   features, then intersect the two cohorts' columns so the pooled matrix is well defined.
5. **Resampling.** SMOTENC is applied to the **training rows only**, so the held-out fold
   contains authentic patient records exclusively.
6. **Range filtering.** `remove_illogical_values` drops scans outside clinically plausible
   ranges, on both partitions, using fixed physiological bounds (not data-derived).
7. **Standardisation.** Continuous features are standardised with statistics estimated on the
   training partition and applied unchanged to the held-out fold.
8. **Hyperparameter selection.** GridSearchCV runs on training and validation folds only.

The country-specific baselines follow the same partitioning: `load_country_dataset` excludes
fold 4 by default, so the baseline fold-models of Figures 3 and 5 are never trained on the rows
they are scored on. `--include-external-fold` restores the older five-fold behaviour.

The baselines are trained on the **ten** shared measurements the Methods enumerate
(`BASELINE_COMMON_FEATURES`). All ten matter: dropping `cpr`, `bpd` or `ute_api` would
understate the reference arm and confound "pooling helps" with "the unified arm had more
predictors". `LEGACY_BASELINE_FEATURES` keeps a seven-feature set
available for reproducing the earlier numbers.

Per-cohort labels are carried through the physiological filter as a marker column rather than
reconstructed from row counts, so dropping a test row cannot silently disable per-cohort
evaluation.

Test rows are always ordered **Malaysia first, then India**, so the `country_arr` indicator
(0 = Malaysia, 1 = India) stays aligned for per-cohort evaluation. If range filtering drops
test rows, `country_arr` is set to `None` rather than silently misaligning.

## Calibration and thresholding

`sga/pipeline/external_fold.py::build_calibrated_external_fold` is the single construction
every externally-reported number uses.

**Which model scores fold 4.** Four fold-models are fitted, one per development fold, each on
the other three folds. Each is scored on its own held-out fold to give a validation AUROC, and
the fold-model with the highest one is the model that scores fold 4 — Methods 2.3.4, "the best
model, judging from the highest AUROC based on the validation sets, was used to evaluate on
the testing data". No model is refitted on the pooled development block. The selected
fold-model carries its own within-feature imputer, cross-domain imputers and scaler onto the
external rows (`prepare_fold(..., external_test_fold=4)`), so those rows are transformed
exactly as its training rows were.

**Which rows calibrate it.** Platt scaling is fitted on the pooled **held-out validation
predictions** of all four fold-models and then applied **unchanged** to the fold-4
probabilities (`sga/evaluation/calibration.py::fit_platt` / `apply_platt`). The fold-4 labels
therefore never enter the calibration map, and the same map defines the probability scale on
which the operating point is chosen.

Those held-out predictions must come from **authentic** rows. SMOTENC is applied to training
rows only, so a fold's held-out rows sit at the cohort's natural prevalence; taking the
out-of-fold predictions over the assembled *training* matrix instead would anchor the Platt
map to a 50% base rate and apply it to a fold at the observed ~21.6%, where it cannot reduce
the calibration error and no threshold chosen on it transfers. `build_calibrated_external_fold`
prints both prevalences on every run and warns when they diverge.

The operating points (Youden's J, and the highest threshold still reaching sensitivity ≥ 0.80)
are chosen on the development block through a **cross-fitted** calibrator
(`platt_cross_fitted`), so no development record is scored by a map its own label helped fit.

Cross-fitting the calibrator *within* fold 4 would be a subtler mistake worth naming: no row
calibrates itself, but the map is still learned from the labels of the samples the metrics are
reported on, which makes ECE and Brier optimistic. `tests/test_manuscript_alignment.py` fails
if any module under `sga/`, `rebuttals/` or `scripts/` fits a calibrator against `test_Y`.

Threshold-dependent metrics (sensitivity, specificity, PPV, NPV) are reported at the default
**0.5** cut-off applied to the calibrated probabilities. The cut-off was not clinically
prespecified nor Youden-selected; a higher-sensitivity alternative is reported separately by
`rebuttals/round2/experiment_R2_3_calibration_uncertainty.py`, with the threshold chosen on
out-of-fold training predictions.

Expected Calibration Error uses **10 equal-width probability bins**
(`sga.config.ECE_BINS`).

## Uncertainty

* Model-performance 95% CIs use the percentile bootstrap with **2,000** iterations
  (`sga.config.N_BOOTSTRAP`).
* Because scans from the same pregnancy remain correlated even under grouped splitting,
  `sga/evaluation/bootstrap.py::bootstrap_metric_ci` accepts `cluster_ids` to resample whole
  pregnancies. `rebuttals/round2/experiment_R2_4_cluster_inference.py` reports both the
  cluster bootstrap and a one-index-scan-per-pregnancy sensitivity analysis.
* AUROC differences between models use DeLong's test, with Holm correction across the
  pairwise comparisons (`sga/evaluation/delong.py::holm_correction`). The correction family is
  the set of model-pair comparisons within one evaluation split of one arm.
  `rebuttals/round1/experiment_R2_9_delong_test.py` writes both the raw `p_value` and the
  adjusted `p_value_holm_bonferroni`; **quote the adjusted one**. `significant` is derived from the
  adjusted value.
* Every stochastic step takes an explicit seed: the resamplers, the ensemble base estimators,
  the permutation-importance shuffles and both bootstrap implementations. A result therefore
  does not depend on how much randomness a preceding stage happened to consume, and a single
  fold can be re-run in isolation.

## Verifying these guarantees

`tests/test_leakage.py` asserts each of the guarantees above against a synthetic cohort pair
built in the real schema, so they can be checked without access to the patient-level data:

```bash
pip install -e ".[dev]"
pytest
```

## What this is, and is not

The unified model is trained on pooled data that **include** the target cohort. This is
**pooled internal validation**, not external validation on an unseen site. The framework
improves performance on the under-represented cohort within the pooled model; it does not
demonstrate transfer to a wholly unseen site. Independent external validation, clinically
justified operating thresholds and cluster-aware uncertainty analysis remain necessary before
any clinical-deployment claim.
