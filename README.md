# A Unified Feature-Harmonization Framework for Small-for-Gestational-Age (SGA) Prediction

Code accompanying *"A unified feature-harmonization framework for small-for-gestational-age
(SGA) prediction: improving cross-border robustness via feature alignment"*.

The framework predicts SGA (birthweight below the 10th INTERGROWTH-21st centile) from
third-trimester antenatal biometry in two cohorts with **different feature spaces** —
Malaysia (n = 18,582; 15 features) and India (n = 1,760; 23 features). A feature-specific
cross-domain imputation stage reconciles the two feature spaces, after which a single
pooled model is trained with a leakage-safe cross-validation and calibration pipeline.

This repository contains everything used to produce the published results and the
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
│   │   └── statistics.py       Table 1 t-test / chi-square / Cohen's d
│   ├── imputation/             The harmonization stage
│   │   ├── registry.py         Feature -> persisted imputer directory
│   │   ├── train_imputers.py   Development-block imputer training + quality metrics
│   │   ├── fold_imputers.py    Per-fold imputer refitting (leakage-safe)
│   │   └── apply.py            impute_df + per-type imputation-quality gating
│   ├── models/
│   │   ├── architecture.py     Feed-forward classifiers (large / medium / small)
│   │   ├── hyperparameters.py  All GridSearchCV grids
│   │   ├── loops.py            DNN train / validate / test epochs
│   │   ├── ensemble.py         Stacking classifier (RF + SVC + LR)
│   │   ├── estimators.py       Grid-searched logistic regression
│   │   ├── domain_adaptation.py CORAL, IRM, DANN comparators
│   │   └── torch_utils.py
│   ├── pipeline/
│   │   ├── dataset.py          Cohort loading, fold train/test construction, scaling
│   │   ├── harmonized_fold.py  prepare_fold() — one harmonized CV fold, end to end
│   │   ├── train_unified.py    Pooled + harmonized training (CatBoost / ML / DNN)
│   │   ├── train_baseline.py   Country-specific baseline training
│   │   ├── inference.py        Held-out fold-4 scoring
│   │   └── model_io.py         Load saved weights, predict, persist predictions
│   ├── evaluation/
│   │   ├── metrics.py          AUROC, AUPRC, ECE, Brier, threshold-dependent metrics
│   │   ├── calibration.py      Platt scaling, reliability curves
│   │   ├── delong.py           DeLong test + Holm correction
│   │   ├── bootstrap.py        Percentile and pregnancy-cluster bootstrap CIs
│   │   ├── fairness.py         Cohort-level disparity metrics
│   │   ├── dca.py              Decision-curve analysis
│   │   └── country.py          Total / Malaysia / India result tables
│   └── reporting/
│       ├── metrics_tables.py, plots.py, artifacts.py
│       ├── importance.py       Permutation importance
│       ├── shap_analysis.py    SHAP rankings (Figure 6)
│       └── figures.py          Manuscript result figures
│
├── scripts/                    Numbered command-line entry points (run in order)
├── tests/                      Pytest suite; runs without the patient-level data
├── rebuttals/
│   ├── round1/                 First set of supplementary analyses (+ README)
│   └── round2/                 Second set of supplementary analyses (+ README)
├── docs/
│   ├── API.md                  Library API reference
│   ├── PIPELINE.md             The leakage-safe pipeline, step by step
│   └── MANUSCRIPT_MAP.md       Every table and figure -> the script that produces it
├── run_all.sh                  Full pipeline, in order
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
git clone <this-repository>
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

Run everything from the repository root. `run_all.sh` executes the whole sequence; the
individual steps are:

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

Rebuttal analyses (run **after** step 3). These produce most of the manuscript's tables:

```bash
# Round 1 -- R0 must run first; it regenerates the imputers and every weight the
# other scripts load. Its __main__ runs inference only, so from a clean checkout
# invoke the retrain phases explicitly first (see the comment in run_all.sh).
python -m rebuttals.round1.experiment_R0_baseline_retrain
python -m rebuttals.round1.experiment_R0_baseline_retrain_manual
python -m rebuttals.round1.experiment_R2_3_additional_metrics    # Tables 2, 4; Figure 4
python -m rebuttals.round1.experiment_R2_9_delong_test           # Table 6
python -m rebuttals.round1.experiment_R2_10_internal_vs_external # Table S3

# Round 2
python -m rebuttals.round2.experiment_R2_3_calibration_uncertainty  # Table 3
python -m rebuttals.round2.experiment_R2_3b_threshold_sweep         # Table S4
python -m rebuttals.round2.experiment_R2_5_fairness_uncertainty     # Table S5, Section 3.9
python -m rebuttals.round2.experiment_R2_4_cluster_inference        # Section 3.8
python -m rebuttals.round2.experiment_R2_1_size_matched_country
# ... see rebuttals/round1/README.md and rebuttals/round2/README.md
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
* Multiplicity across the pairwise DeLong tests is controlled by the **Holm-Bonferroni**
  step-down procedure (`p_value_holm_bonferroni`), verified against `statsmodels`.
* Confidence intervals follow the method the Statistical Analysis section specifies for
  each quantity: **Wilson score** intervals for proportions (sensitivity, specificity, PPV,
  NPV, cohort-level TPR/FPR), **Newcombe hybrid score** intervals for differences between two
  independent proportions, and a percentile **bootstrap** (2,000 iterations) for AUROC, AUPRC,
  ECE and Brier. The pregnancy-cluster bootstrap resamples pregnancies rather than scans.
  `sga/config.py::CI_METHODS` records the mapping and the test suite enforces it.
* The external test fold is built once, by `sga/pipeline/external_fold.py`, and shared by every
  script that reports it, so the calibration table, the threshold sweep, the cluster analysis
  and the fairness section cannot drift apart.
* Every stochastic step takes an explicit seed, so results do not depend on execution order
  and any single fold can be re-run on its own.
* `pytest` asserts the leakage guarantees above on synthetic data, without the private cohort.

See [`docs/PIPELINE.md`](docs/PIPELINE.md) for the full fold-by-fold sequence.

---

## 6. Where each result comes from

[`docs/MANUSCRIPT_MAP.md`](docs/MANUSCRIPT_MAP.md) maps every table, figure and quoted
number to the script that produces it, the file it writes and the column to read.

---

## 7. Citation

```bibtex
@article{lim_sga_harmonization,
  title   = {A unified feature-harmonization framework for small-for-gestational-age (SGA)
             prediction: improving cross-border robustness via feature alignment},
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
