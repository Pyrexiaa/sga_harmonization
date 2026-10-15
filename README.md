# Cross-Site Pooling with Feature Harmonization for Small-for-Gestational-Age (SGA) Prediction

The official code accompanying *"Cross-site pooling with feature harmonization for small-for-gestational-age
(SGA) prediction across heterogeneous feature spaces"*.

**Version 1.0.0** — the release used for the paper.

The framework predicts SGA (birthweight below the 10th INTERGROWTH-21st centile) from
third-trimester antenatal biometry in two cohorts with **different feature spaces** —
Malaysia (n = 18,582; 15 features) and India (n = 1,760; 23 features). A feature-specific
cross-domain imputation stage reconciles the two feature spaces, after which a single
pooled model is trained with a leakage-safe cross-validation and calibration pipeline. Of the
candidate cross-domain features, only uterine artery resistance index (`ute_ari`) meets the
imputation-quality criteria, so the harmonized feature set is the ten common predictors plus
`ute_ari`.

This repository contains everything used to produce the reported results and the
supplementary analyses: data preparation, imputation-model training, classifier training and
inference, evaluation, and figure generation.

> **Not for clinical use.** The models here are research artefacts. The associated web
> demonstrator has no regulatory clearance and is not intended to guide patient management.

---

## 1. Repository layout

```
sga_harmonization/
├── sga/                        Importable library — all shared logic lives here
│   ├── config.py               Seeds, paths, feature lists, thresholds (single source of truth)
│   ├── data/                   Cohort preparation
│   │   ├── preprocess_malaysia.py, preprocess_india.py
│   │   ├── screening.py        CONSORT-style eligibility accounting
│   │   ├── centiles.py         INTERGROWTH-21st / Hadlock / WHO centile lookup, SGA labelling
│   │   ├── cleaning.py         Physiological range filtering, iterative imputation
│   │   ├── encoding.py         Categorical encoding
│   │   ├── scaling.py          Train-fitted standardisation
│   │   ├── splits.py           Patient-grouped stratified k-fold assignment
│   │   └── statistics.py       Table 1 tests; SMD and overlap (Tables S3, S5)
│   ├── imputation/             The harmonization stage
│   │   ├── registry.py         Feature -> persisted imputer directory
│   │   ├── train_imputers.py   Development-block imputer training + quality metrics
│   │   ├── fold_imputers.py    Per-fold imputer refitting (leakage-safe)
│   │   ├── diagnostics.py      Multiclass imputer diagnostics (Table S2c)
│   │   └── apply.py            impute_df + per-type imputation-quality gating
│   ├── models/
│   │   ├── architecture.py     Feed-forward classifiers (large / medium / small)
│   │   ├── hyperparameters.py  Search grids (Table S11)
│   │   ├── loops.py            DNN train / validate / test epochs
│   │   ├── ensemble.py         Stacking classifier (RF + SVC + LR)
│   │   ├── estimators.py       Grid-searched LR / RF / SVC / CatBoost
│   │   ├── domain_adaptation.py CORAL, IRM, DANN comparators
│   │   └── torch_utils.py
│   ├── pipeline/
│   │   ├── dataset.py          Cohort loading, fold train/test construction, scaling
│   │   ├── harmonized_fold.py  prepare_fold() — one harmonized CV fold, end to end
│   │   ├── train_unified.py    Pooled + harmonized training (CatBoost / ML / DNN)
│   │   ├── train_baseline.py   Country-specific baseline training
│   │   ├── external_fold.py    The calibrated fold-4 evaluation every reported number uses
│   │   ├── incomplete_records.py  Out-of-fold scoring of every record (Table S10)
│   │   ├── inference.py        Held-out fold-4 scoring
│   │   └── model_io.py         Load saved weights, predict, persist predictions
│   ├── evaluation/
│   │   ├── metrics.py          AUROC, AUPRC, ECE, Brier, threshold-dependent metrics
│   │   ├── calibration.py      Platt scaling, reliability curves
│   │   ├── delong.py           DeLong test + Holm correction
│   │   ├── bootstrap.py        Percentile, stratified and pregnancy-cluster bootstrap CIs
│   │   ├── proportions.py      Wilson and Newcombe score intervals
│   │   ├── comparator.py       EFW < 10th centile rule as a comparator (Table S9)
│   │   ├── fairness.py         Cohort-level disparity metrics
│   │   ├── dca.py              Decision-curve analysis
│   │   └── country.py          Total / Malaysia / India result tables
│   └── reporting/
│       ├── metrics_tables.py, plots.py, artifacts.py
│       ├── importance.py       Permutation importance
│       ├── shap_analysis.py    SHAP rankings (Figure 6)
│       └── figures.py          Result figures
│
├── scripts/                    Numbered command-line entry points (run in order)
├── tests/                      Pytest suite; runs without the patient-level data
├── rebuttals/
│   ├── round1/                 Supplementary analyses, set 1 (+ README)
│   ├── round2/                 Supplementary analyses, set 2 (+ README)
│   └── round3/                 Supplementary analyses, set 3 (+ README)
├── docs/
│   ├── API.md                  Library API reference
│   └── PIPELINE.md             The leakage-safe pipeline, step by step
├── run_all.sh                  Full pipeline, in order
├── CITATION.cff                Citation metadata for this software
├── LICENSE
├── .gitignore
├── requirements.txt / environment.yml / pyproject.toml
└── README.md
```

**Design rule:** nothing analytical lives in a script. Scripts under `scripts/` and
`rebuttals/` are thin entry points; every metric, fold, imputation and plot is a function
in `sga/`. If you need a helper, it is in [`docs/API.md`](docs/API.md) — do not re-implement it.

---

## 2. Installation

```bash
git clone https://github.com/Pyrexiaa/sga_harmonization.git
cd sga_harmonization

# Option A — pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"      # [dev] adds pytest

# Option B — conda
conda env create -f environment.yml
conda activate sga-harmonization
pip install -e .
```

Python 3.10-3.12. A GPU is optional; CatBoost and PyTorch fall back to CPU automatically.

Verify the install with the test suite, which needs **no** patient-level data — it builds a
synthetic cohort pair in the same schema:

```bash
pytest
```

---

## 3. Data you need to supply

Patient-level data cannot be redistributed and is **available from the corresponding author
on request**. The code expects the following directories relative to the project root
(override with the `SGA_PROJECT_ROOT` environment variable):

| Path | Contents |
|---|---|
| `Datasets/RawDatasets/` | Raw Malaysia and India exports, before screening |
| `RefCentile/` | `I21_BW.csv`, `EFW_Centile.xlsx` and, optionally, `HL_EFW.csv`, `WHO_EFW.csv`, `MSIA_EFW.csv` |
| `Datasets/FinalDatasetsForTraining_123/` | Written by `scripts/01a`/`01b`: `Malaysia/` and `India/`, each with `tri3_i21.csv` and `tri3_add_on_i21.csv` |
| `Results/` | Everything the pipeline writes |

`tri3_i21.csv` holds the complete-case records with a `fold` column (0-4);
`tri3_add_on_i21.csv` holds records that still contain missing values, labelled `fold = -1`.
Add-on records only ever enter a **training** partition.

---

## 4. Running the pipeline

Run everything from the repository root. `bash run_all.sh` executes the whole sequence, and
`bash run_all.sh <stage> ...` runs selected stages (`prepare`, `impute`, `train`, `test`,
`rebuttal1`, `figures`, `rebuttal2`, `rebuttal3`). The individual steps are:

```bash
# 1. Prepare the cohorts (screening, SGA labelling, patient-grouped folds)
python scripts/01a_prepare_malaysia.py
python scripts/01b_prepare_india.py
python scripts/01c_cohort_characteristics.py          # Table 1

# 2. Train the cross-domain feature imputers (the harmonization stage)
python scripts/02_train_imputers.py

# 3. Train the classifiers
#    (a) unified harmonized: common features + the retained cross-domain features
python scripts/03a_train_unified_catboost.py
python scripts/03b_train_unified_ml.py --model lr     # also: rf, svc, stacking
python scripts/03c_train_unified_dnn.py

#    (b) unified common-feature only -- the no-cross-domain-imputation arm of the
#        Table 7 ablation, NOT one of Figure 3's three arms (those are baseline /
#        unified / cross-domain). `--selected-features` with no values selects the
#        empty cross-domain set.
python scripts/03b_train_unified_ml.py --model lr --selected-features \
    --output-dir Results/models/unified_common_lr_123

#    (c) country-specific baselines -- Figure 3's left arm and both Figure 5
#        baseline arms. Every classifier needs its own baseline, not just CatBoost.
python scripts/03d_train_country_baseline.py --country malaysia --model lr
python scripts/03d_train_country_baseline.py --country india --model lr

# 4. Score the held-out external fold (fold 4). Every arm the figures read must be
#    scored on the SAME held-out rows.
python scripts/04a_test_unified_catboost.py
python scripts/04b_test_unified_ml.py --model lr
python scripts/04c_test_unified_dnn.py
python scripts/04b_test_unified_ml.py --model lr --selected-features \
    --model-dir Results/models/unified_common_lr_123
python scripts/04d_test_country_baseline.py --country malaysia --model lr
python scripts/04d_test_country_baseline.py --country india --model lr

# 5. Figures
python scripts/05a_figure3_auroc_comparison.py        # Figure 3

# 05b0 re-scores one arm at a time on the identical held-out rows, so it takes an
# explicit --model-dir. The baseline arms additionally need --baseline-features,
# because their weights were fitted on the common-feature space only.
python scripts/05b0_subgroup_inference.py --family ml --model lr \
    --model-dir Results/models/unified_lr_123 --name unified
python scripts/05b0_subgroup_inference.py --family ml --model lr \
    --model-dir Results/models/baseline_malaysia_lr_123 \
    --train-source malaysia --baseline-features --name malaysia_baseline
python scripts/05b0_subgroup_inference.py --family ml --model lr \
    --model-dir Results/models/baseline_india_lr_123 \
    --train-source india --baseline-features --name india_baseline
python scripts/05b_figure5_subgroup_analysis.py       # Figure 5
python scripts/05c_figure7_sample_size.py             # Figure 7
```

Supplementary analyses under `rebuttals/` (run **after** step 3). These produce most of the
tables:

```bash
# Round 1 -- R0 must run first; it regenerates the imputers and every weight the
# other scripts load. Its __main__ runs inference only, so from a clean checkout
# invoke the retrain phases explicitly first (see the comment in run_all.sh).
python -m rebuttals.round1.experiment_R0_baseline_retrain
python -m rebuttals.round1.experiment_R0_baseline_retrain_manual
python -m rebuttals.round1.experiment_R2_3_additional_metrics    # Tables 2, 4; Figure 4
python -m rebuttals.round1.experiment_R2_9_delong_test           # Table 6
python -m rebuttals.round1.experiment_R2_10_internal_vs_external # Table S6

# Round 2
python -m rebuttals.round2.experiment_R2_3_calibration_uncertainty  # Table 3
python -m rebuttals.round2.experiment_R2_3b_threshold_sweep         # Table S4
python -m rebuttals.round2.experiment_R2_5_fairness_uncertainty     # Table S7, Section 3.9
python -m rebuttals.round2.experiment_R2_4_cluster_inference        # Section 3.8
python -m rebuttals.round2.experiment_R2_1_size_matched_country

# Round 3
python -m rebuttals.round3.experiment_R2_2_af_diagnostic            # Table S2c
python -m rebuttals.round3.experiment_R2_2b_af_downstream           # Table 7a, af arm
python -m rebuttals.round3.experiment_R3_6_cohort_shift_tables      # Tables S3, S5
python -m rebuttals.round3.experiment_R3_5_efw_centile_comparator   # Table S9, Section 3.5
python -m rebuttals.round3.experiment_R3_1_incomplete_records       # Table S10, Section 3.3
# ... see the README in each rebuttals/round* directory
```

---

## 5. Reproducibility

* The global seed is `123`, set in `sga/config.py` and applied to Python, NumPy and PyTorch
  at the top of every entry point. Override with `SGA_SEED=<n>`.
* Fold assignment is **patient-grouped**: all third-trimester scans from one pregnancy fall
  in the same fold. Folds 0-3 are the development block; **fold 4 is the held-out external
  test partition** and is never seen during imputer fitting, feature selection, resampling,
  hyperparameter tuning, calibration or thresholding.
* Both imputation stages are **fitted inside every fold**, on that fold's training rows only:
  the cross-domain feature imputers and the within-feature iterative imputer. With four
  development folds the fold-3 imputers are fitted on folds 0-2 and applied to fold 3. Reusing
  one globally fitted imputer would let a held-out row shape the model that fills its own
  fold's training data.
* Add-on (incomplete) records are matched on pregnancy `id` against both the external fold and
  the current CV fold, so no second scan of a held-out pregnancy reaches training.
* Platt scaling is fitted on the folds 0-3 **held-out** predictions - authentic rows at the
  cohort's natural prevalence, since oversampling touches training rows only - and applied
  unchanged to fold 4, so the reported test labels never inform the calibration map.
* Fold 4 is scored by the development fold-model with the highest validation AUROC, carrying
  that fold's own imputers and scaler; no model is refitted on the pooled development block.
* Hyperparameters are chosen by grid search with AUROC as the metric, using the settings
  of appendix Table S11 (`sga.config.GRID_SEARCH_CV_FOLDS`, `LR_REFIT_CV_FOLDS`,
  `CATBOOST_BOOSTING_TYPES`, `CATBOOST_EARLY_STOPPING_ROUNDS`).
* Multiplicity across the pairwise DeLong tests is controlled by the **Holm-Bonferroni**
  step-down procedure (`p_value_holm_bonferroni`), verified against `statsmodels`.
* Confidence intervals: **Wilson score** intervals for proportions (sensitivity, specificity, PPV,
  NPV, cohort-level TPR/FPR), **Newcombe hybrid score** intervals for differences between two
  independent proportions, and a percentile **bootstrap** (2,000 iterations) for AUROC, AUPRC,
  ECE and Brier. Each reported interval states its resampling unit: pregnancies for the
  cluster-aware analyses, scans within outcome classes for Figure 3 and scans within cohorts
  for Table 3.
  `sga/config.py::CI_METHODS` records which method applies to which quantity.
* The calibrated external test fold is built once, by `sga/pipeline/external_fold.py`, and
  shared by every script that reports it, so the calibration table, the threshold sweep, the
  cluster analysis, the fairness analysis and the EFW-centile comparison use the same model,
  calibrator and rows.
* Every stochastic step takes an explicit seed, so results do not depend on execution order
  and any single fold can be re-run on its own.
* `pytest` asserts the leakage guarantees above on synthetic data, without the private cohort.

See [`docs/PIPELINE.md`](docs/PIPELINE.md) for the full fold-by-fold sequence.

---

## 6. Where each result comes from

`R1`, `R2` and `R3` stand for `rebuttals.round1`, `rebuttals.round2` and `rebuttals.round3`
(run as `python -m rebuttals.round1.<script>`); their outputs land under
`Results/rebuttal_round{1,2,3}/`. Each `rebuttals/round*/README.md` lists the files written.

| Result | Produced by |
|---|---|
| Table 1, Table S1 | `scripts/01c_cohort_characteristics.py` |
| Tables 2 and 4, Figure 4 | `R1.experiment_R2_3_additional_metrics` |
| Table 3 and the operating points in §3.2 | `R2.experiment_R2_3_calibration_uncertainty` |
| Table 5 | `R1.experiment_R2_1_domain_generalization_comparison` |
| Table 6 | `R1.experiment_R2_9_delong_test` |
| Table 7 | `R1.experiment_R2_5_imputation_ablation`; the amniotic-fluid arm `R3.experiment_R2_2b_af_downstream` |
| Table 8 | `R1.experiment_R2_6_weighted_training` |
| Figure 3 | `scripts/05a_figure3_auroc_comparison.py` |
| Figure 5 | `scripts/05b0_subgroup_inference.py` (once per arm), then `scripts/05b_figure5_subgroup_analysis.py` |
| Figure 6 | `R1.shap_without_retraining` |
| Figure 7 | `R1.experiment_R1_2_data_scaling`, `R1.experiment_R1_2_data_scaling_inference`, then `scripts/05c_figure7_sample_size.py` |
| Tables S2a, S2b | `R1.experiment_R2_2_imputation_metrics` |
| Table S2c | `R3.experiment_R2_2_af_diagnostic` |
| Tables S3, S5 | `R3.experiment_R3_6_cohort_shift_tables` |
| Table S4 | `R2.experiment_R2_3b_threshold_sweep` |
| Table S6 | `R1.experiment_R2_10_internal_vs_external` |
| Table S7 and §3.9 | `R2.experiment_R2_5_fairness_uncertainty` |
| Table S8 | `R1.experiment_R2_1b_comparator_settings` |
| Table S9 and §3.5 | `R3.experiment_R3_5_efw_centile_comparator` |
| Table S10 | `R3.experiment_R3_1_incomplete_records` |
| Table S11 | `sga/models/hyperparameters.py`, `sga/models/estimators.py` |
| §3.2 size-matched analysis | `R2.experiment_R2_1_size_matched_country` |
| §3.8 pregnancy-cluster analysis | `R2.experiment_R2_4_cluster_inference` |
| India AUROC gain, per-fold DeLong | `R2.experiment_R3_2_india_auroc_delong` |
| AUPRC gain by cohort (imputed vs native `ute_ari`) | `R2.experiment_R3_5_imputed_vs_native` |

---

## 7. Citation

If you use this code, please cite the paper. Citation metadata for the software itself is in
[`CITATION.cff`](CITATION.cff) (GitHub shows it under "Cite this repository").

```bibtex
@article{lim_sga_harmonization,
  title   = {Cross-site pooling with feature harmonization for small-for-gestational-age (SGA)
             prediction across heterogeneous feature spaces},
  author  = {Lim, Jia Yu and Nirmalan, Praveen Kumar and Choorakuttil, Rijo Mathew and
             Sethi, Neha Sethi A/P Naresh and Kamar, Azanna Ahmad and Saaid, Rahmah and
             Jalil, Nurul Syazwani and Ng, Kwan Hoong and Saw, Shier Nee},
  year    = {2026}
}
```

## 8. Ethics and funding

Approved by the Medical Research Ethics Committee, University Malaya Medical Centre
(ID 2021329-9997) and the AMMA Center for Diagnosis and Preventive Medicine Pvt Ltd, Kochi
(Ethical Committee 05/2023). Informed consent was waived by both committees. Supported in
part by the Ministry of Higher Education Malaysia, Fundamental Research Grant Scheme
(FRGS/1/2023/SKK05/UM/02/3).

## 9. Licence

MIT — see [`LICENSE`](LICENSE).
