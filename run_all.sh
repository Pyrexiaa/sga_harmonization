#!/usr/bin/env bash
#
# Full pipeline for the unified feature-harmonization SGA framework.
# Run from the repository root, with the environment already activated:
#
#   bash run_all.sh                 # everything
#   bash run_all.sh prepare train   # selected stages only
#
# Stages: prepare | impute | train | test | figures | rebuttal1 | rebuttal2
#
# Override the project root (where Datasets/, RefCentile/ and Results/ live)
# and the global seed with environment variables:
#   SGA_PROJECT_ROOT=/data/sga SGA_SEED=123 bash run_all.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
SEED="${SGA_SEED:-123}"
PROJECT_ROOT="${SGA_PROJECT_ROOT:-$(pwd)}"
MODEL_ROOT="$PROJECT_ROOT/Results/models"
STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
    # `figures` runs AFTER `rebuttal1`: Figure 3's cross-domain arm and Figure 7's
    # size sweep are both produced by the round-1 tree, and 05a/05c abort without
    # them. With `set -euo pipefail` the old order aborted every clean run.
    STAGES=(prepare impute train test rebuttal1 figures rebuttal2)
fi

has_stage() {
    local needle=$1
    for stage in "${STAGES[@]}"; do
        [ "$stage" = "$needle" ] && return 0
    done
    return 1
}

banner() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

if has_stage prepare; then
    banner "1. Cohort preparation"
    $PYTHON scripts/01a_prepare_malaysia.py
    $PYTHON scripts/01b_prepare_india.py
    $PYTHON scripts/01c_cohort_characteristics.py        # Table 1
fi

if has_stage impute; then
    banner "2. Cross-domain feature imputers"
    $PYTHON scripts/02_train_imputers.py
fi

if has_stage train; then
    banner "3a. Unified harmonized models (common + cross-domain features)"
    $PYTHON scripts/03a_train_unified_catboost.py
    for model in lr rf svc stacking; do
        $PYTHON scripts/03b_train_unified_ml.py --model "$model"
    done
    $PYTHON scripts/03c_train_unified_dnn.py

    # Pooled training on the COMMON features only (no cross-domain imputation).
    # This is the ablation that isolates the contribution of the two imputed
    # features; Figure 3's "unified" arm is the harmonized run from stage 3a.
    banner "3b. Unified common-feature models (imputation ablation)"
    $PYTHON scripts/03a_train_unified_catboost.py --selected-features \
        --output-dir "$MODEL_ROOT/unified_common_catboost_$SEED"
    for model in lr rf svc stacking; do
        $PYTHON scripts/03b_train_unified_ml.py --model "$model" --selected-features \
            --output-dir "$MODEL_ROOT/unified_common_${model}_$SEED"
    done
    $PYTHON scripts/03c_train_unified_dnn.py --selected-features \
        --output-dir "$MODEL_ROOT/unified_common_dnn_$SEED"

    # Figure 3's left arm and both baseline arms of Figure 5. Every classifier
    # needs its own country baseline, not just CatBoost. Trained on the ten
    # common features, as the Methods specify.
    banner "3c. Country-specific baselines (Figure 3, 'baseline' arm)"
    for country in malaysia india; do
        for model in catboost lr rf svc stacking dnn; do
            $PYTHON scripts/03d_train_country_baseline.py \
                --country "$country" --model "$model"
        done
    done
fi

if has_stage test; then
    banner "4. Held-out external fold (fold 4)"
    # Every arm that Figure 3 reads must be scored on the SAME held-out rows.
    $PYTHON scripts/04a_test_unified_catboost.py
    for model in lr rf svc stacking; do
        $PYTHON scripts/04b_test_unified_ml.py --model "$model"
    done
    $PYTHON scripts/04c_test_unified_dnn.py

    $PYTHON scripts/04a_test_unified_catboost.py --selected-features \
        --model-dir "$MODEL_ROOT/unified_common_catboost_$SEED"
    for model in lr rf svc stacking; do
        $PYTHON scripts/04b_test_unified_ml.py --model "$model" --selected-features \
            --model-dir "$MODEL_ROOT/unified_common_${model}_$SEED"
    done
    $PYTHON scripts/04c_test_unified_dnn.py --selected-features \
        --model-dir "$MODEL_ROOT/unified_common_dnn_$SEED"

    for country in malaysia india; do
        for model in catboost lr rf svc stacking dnn; do
            $PYTHON scripts/04d_test_country_baseline.py \
                --country "$country" --model "$model"
        done
    done
fi

if has_stage figures; then
    banner "5. Manuscript figures"
    # Figure 3 has THREE arms per cohort. The baseline and unified arms come from
    # stages 3-4 above; the cross-domain arm (trained on the OTHER cohort, ten common
    # features) is produced by the round-1 tree, so `rebuttal1` must have run at least
    # once. 05a now fails loudly rather than quietly drawing a two-armed figure --
    # pass --allow-missing-strategies if a partial panel is genuinely what you want.
    $PYTHON scripts/05a_figure3_auroc_comparison.py      # Figure 3

    # Figure 5 compares three arms on identical rows, so 05b0 runs once per arm
    # with an explicit --model-dir. It has no usable defaults.
    $PYTHON scripts/05b0_subgroup_inference.py --family ml --model lr \
        --model-dir "$MODEL_ROOT/unified_lr_$SEED" --name unified
    $PYTHON scripts/05b0_subgroup_inference.py --family ml --model lr \
        --model-dir "$MODEL_ROOT/baseline_malaysia_lr_$SEED" \
        --train-source malaysia --baseline-features --name malaysia_baseline
    $PYTHON scripts/05b0_subgroup_inference.py --family ml --model lr \
        --model-dir "$MODEL_ROOT/baseline_india_lr_$SEED" \
        --train-source india --baseline-features --name india_baseline
    $PYTHON scripts/05b_figure5_subgroup_analysis.py     # Figure 5
    $PYTHON scripts/05c_figure7_sample_size.py           # Figure 7
fi

if has_stage rebuttal1; then
    banner "6. Round-1 rebuttal analyses"
    # NOTE: `experiment_R0_baseline_retrain` as a module only runs INFERENCE over
    # weights that already exist -- its retraining phases are deliberately not
    # invoked from __main__ because they take days on a GPU node. From a clean
    # checkout you must first run them explicitly, e.g.
    #
    #   python -c "from rebuttals.round1.experiment_R0_baseline_retrain import *; \
    #              retrain_imputation_models(); retrain_unified_catboost(); \
    #              retrain_unified_ml(); retrain_unified_dnn(); \
    #              retrain_single_source_all(); retrain_single_source_common_all()"
    #
    # Figure 3's cross-domain arm comes from retrain_single_source_common_all();
    # `experiment_R0_baseline_retrain` then scores those weights and writes each run's
    # external_fold4_per_fold_results.csv where scripts/05a reads it.
    $PYTHON -m rebuttals.round1.experiment_R0_baseline_retrain
    $PYTHON -m rebuttals.round1.experiment_R0_baseline_retrain_manual
    for experiment in \
        experiment_R1_1_remove_prev_pregnancy \
        experiment_R1_2_data_scaling \
        experiment_R1_2_data_scaling_inference \
        experiment_R2_1_domain_generalization_comparison \
        experiment_R2_1b_comparator_settings \
        experiment_R2_2_imputation_metrics \
        experiment_R2_3_additional_metrics \
        experiment_R2_4_decision_curve_analysis \
        experiment_R2_5_imputation_ablation \
        experiment_R2_6_weighted_training \
        experiment_R2_7_split_unified_by_country \
        experiment_R2_8_fairness_metrics \
        experiment_R2_9_delong_test \
        experiment_R2_10_internal_vs_external \
        evaluate_per_country \
        shap_without_retraining
    do
        $PYTHON -m "rebuttals.round1.$experiment"
    done
fi

if has_stage rebuttal2; then
    banner "7. Round-2 rebuttal analyses"
    for experiment in \
        experiment_R2_1_size_matched_country \
        experiment_R2_3_calibration_uncertainty \
        experiment_R2_3b_threshold_sweep \
        experiment_R2_4_cluster_inference \
        experiment_R2_5_fairness_uncertainty \
        experiment_R3_2_india_auroc_delong \
        experiment_R3_5_imputed_vs_native
    do
        $PYTHON -m "rebuttals.round2.$experiment"
    done
fi

banner "Done"
