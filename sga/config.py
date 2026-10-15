"""Central configuration for the unified feature-harmonization SGA framework."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = int(os.environ.get("SGA_SEED", 123))


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (when available) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Paths ────────────────────────────────────────────────────────────────────
def _default_project_root() -> Path:
    """Repository root, independent of the process working directory."""
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.cwd()


PROJECT_ROOT = Path(
    os.environ.get("SGA_PROJECT_ROOT") or _default_project_root()
).resolve()

RAW_DATA_DIR = PROJECT_ROOT / "Datasets" / "RawDatasets"
PROCESSED_DATA_DIR = PROJECT_ROOT / "Datasets" / "ProcessedDatasets"
TRAINING_DATA_DIR = PROJECT_ROOT / "Datasets" / f"FinalDatasetsForTraining_{SEED}"
CENTILE_DIR = PROJECT_ROOT / "RefCentile"

RESULTS_DIR = PROJECT_ROOT / "Results"
MODEL_DIR = RESULTS_DIR / "models"
IMPUTER_DIR = MODEL_DIR / f"catboost_feature_imputers_{SEED}"
FIGURE_DIR = RESULTS_DIR / "figures"
ROUND1_DIR = RESULTS_DIR / "rebuttal_round1"
ROUND2_DIR = RESULTS_DIR / "rebuttal_round2"
ROUND3_DIR = RESULTS_DIR / "rebuttal_round3"
TRANSFER_DIR = RESULTS_DIR / "transfer_learning"

MALAYSIA_SUBDIR = "Malaysia"
INDIA_SUBDIR = "India"

#: Cohort codes used in every ``country_arr``. Malaysia is 0 and comes first in
#: the concatenated test block; India is 1 and comes second.
MALAYSIA = 0
INDIA = 1

# Birthweight reference chart used for the SGA label (INTERGROWTH-21st).
CHART = "i21"
LABEL = "sga"

# ── Cross-validation ─────────────────────────────────────────────────────────
# Five patient-grouped stratified folds: 0-3 for development (four-fold
# cross-validation), 4 held out as the external test partition.
N_FOLDS_TOTAL = 5
N_FOLDS_CV = 4
EXTERNAL_TEST_FOLD = 4

# ── Features ─────────────────────────────────────────────────────────────────
# The ten features measured in BOTH cohorts.
COMMON_FEATURES = [
    "m_age",     # maternal age
    "gender",    # fetal sex
    "hc",        # head circumference
    "ac",        # abdominal circumference
    "fl",        # femur length
    "efw",       # estimated fetal weight
    "ga",        # gestational age at scan
    "cpr",       # cerebroplacental ratio
    "bpd",       # biparietal diameter
    "ute_api",   # uterine artery pulsatility index
]

# Candidate cross-domain features, grouped by the imputation model type used.
MALAYSIA_MULTICLASS_FEATURES = ["af", "placenta_site"]
MALAYSIA_REGRESSION_FEATURES = ["afi", "psv", "ute_ari"]
INDIA_BINARY_FEATURES = [
    "hypertension_0",
    "hypertension_1",
    "diabetes_0",
    "diabetes_1",
    "smoking",
    "last_preg_sga",
    "last_preg_fgr",
    "last_preg_normal",
    "prev_failed_preg",
    "high_risk_pe",
]
INDIA_REGRESSION_FEATURES = ["umb_api", "m_height", "m_weight"]

ALL_CROSS_DOMAIN_FEATURES = (
    MALAYSIA_MULTICLASS_FEATURES
    + MALAYSIA_REGRESSION_FEATURES
    + INDIA_BINARY_FEATURES
    + INDIA_REGRESSION_FEATURES
)

# Maternal-history features available only in India.
PREV_PREGNANCY_FEATURES = [
    "last_preg_sga",
    "last_preg_fgr",
    "last_preg_normal",
    "prev_failed_preg",
    "high_risk_pe",
]

# The only cross-domain feature that met the retention criteria (appendix Table S2);
# amniotic fluid condition fails the multiclass criteria (Table S2c) and is excluded.
HARMONIZED_SELECTED_FEATURES = ["ute_ari"]

# Features never imputed: they are absent from Malaysia and constant in India.
CONSTANT_ZERO_FEATURES = ["smoking", "diabetes_1"]

CATEGORICAL_FEATURES = set(MALAYSIA_MULTICLASS_FEATURES) | set(INDIA_BINARY_FEATURES)
MALAYSIA_CATEGORICAL = ["gender", "mode_of_delivery", "admission_place"]
# ``high_risk_pe`` and ``prev_failed_preg`` are already in INDIA_BINARY_FEATURES.
INDIA_CATEGORICAL = ["gender"] + list(INDIA_BINARY_FEATURES)

# ── Imputation-quality retention thresholds ─────────────────────────────────
# Every bound is strict (appendix Table S2):
#   binary      AUROC > 0.80 and F1 > 0.50
#   multiclass  F1 > 0.80, AUROC > 0.50, weighted F1 above the always-predict-majority
#               baseline, and a macro one-vs-rest AUROC 95% CI that excludes 0.50
#   continuous  R^2 > 0.80, with MAE inspected as a complementary measure
BINARY_AUROC_THRESHOLD = 0.80
BINARY_F1_THRESHOLD = 0.50
MULTICLASS_F1_THRESHOLD = 0.80
MULTICLASS_AUROC_THRESHOLD = 0.50
REGRESSION_R2_THRESHOLD = 0.80

#: Primary cut-off passed by the training scripts; it equals every per-type primary
#: threshold above, so the retained set is the same as with the per-type constants.
ACCURACY_THRESHOLD = 0.8

# ── Hyperparameter search (appendix Table S11) ───────────────────────────────
#: Inner cross-validation folds of the six-classifier grid searches.
GRID_SEARCH_CV_FOLDS = 3
#: Inner folds of the Logistic Regression refitted for the calibration, operating
#: threshold and fairness analyses (Tables 3, S4 and S7).
LR_REFIT_CV_FOLDS = 5
#: CatBoost boosting types searched in the reported run.
CATBOOST_BOOSTING_TYPES = ["Plain"]
#: CatBoost stops after this many rounds without improvement on the fold's
#: held-out validation rows.
CATBOOST_EARLY_STOPPING_ROUNDS = 50

# ── Evaluation ───────────────────────────────────────────────────────────────
DECISION_THRESHOLD = 0.5      # default operating point for threshold-dependent metrics
ECE_BINS = 10                 # equal-width probability bins for ECE
N_BOOTSTRAP = 2000            # bootstrap iterations for the AUROC/AUPRC/ECE/Brier CIs
ALPHA = 0.05
DCA_THRESHOLD_RANGE = (0.0, 0.5)   # decision-curve threshold probabilities
DCA_CLINICAL_RANGE = (0.05, 0.20)  # clinically relevant range highlighted in Fig. 4

#: Conventional clinical rule used as the comparator in appendix Table S9.
EFW_CENTILE_RULE = 10.0

#: Cut-offs tabulated in appendix Tables S4 (sensitivity/specificity) and S7
#: (cohort-level fairness): 0.10 to 0.90 in steps of 0.05.
THRESHOLD_GRID = [round(0.10 + 0.05 * step, 2) for step in range(17)]

#: Screening operating point used in the Cohort-Level Fairness section, chosen a
#: priori as representative of the 5-20% clinically relevant range that the
#: decision-curve analysis identifies. It is NOT tuned on the test fold.
SCREENING_THRESHOLD = 0.10

#: Below this many positive predictions in a cohort, PPV - and therefore the
#: predictive-parity difference - is reported but flagged as not reliably
#: estimable (the "*" footnote of appendix Table S7).
MIN_POSITIVE_PREDICTIONS_FOR_PPV = 10

#: Confidence-interval method per quantity. Proportions and their differences use closed-form score intervals;
#: only the non-proportion metrics are bootstrapped.
CI_METHODS = {
    "proportion": "Wilson score",
    "proportion_difference": "Newcombe hybrid score",
    "auroc": "percentile bootstrap",
    "auprc": "percentile bootstrap",
    "ece": "percentile bootstrap",
    "brier": "percentile bootstrap",
}

MODEL_DISPLAY_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "svc": "Support Vector Classifier",
    "stacking": "Stacking Classifier",
    "catboost": "CatBoost",
    "dnn": "Neural Network",
}


def results_path(*parts) -> Path:
    """Build (and create) a directory under the results root."""
    path = RESULTS_DIR.joinpath(*[str(p) for p in parts])
    path.mkdir(parents=True, exist_ok=True)
    return path
