# `sga` library API reference

Import everything from the `sga` package; never re-implement a helper listed here.

## `sga.config`
Constants: `SEED`, `PROJECT_ROOT`, `TRAINING_DATA_DIR`, `CENTILE_DIR`, `RESULTS_DIR`,
`MODEL_DIR`, `IMPUTER_DIR`, `FIGURE_DIR`, `ROUND1_DIR`, `ROUND2_DIR`, `TRANSFER_DIR`,
`MALAYSIA_SUBDIR` (`"Malaysia"`), `INDIA_SUBDIR` (`"India"`), `CHART` (`"i21"`), `LABEL` (`"sga"`),
`N_FOLDS_TOTAL` (5), `N_FOLDS_CV` (4), `EXTERNAL_TEST_FOLD` (4),
`COMMON_FEATURES` (the 10 shared features), `MALAYSIA_MULTICLASS_FEATURES`,
`MALAYSIA_REGRESSION_FEATURES`, `INDIA_BINARY_FEATURES`, `INDIA_REGRESSION_FEATURES`,
`ALL_CROSS_DOMAIN_FEATURES`, `PREV_PREGNANCY_FEATURES`, `HARMONIZED_SELECTED_FEATURES`
(`["ute_ari", "af"]`), `CONSTANT_ZERO_FEATURES`, `CATEGORICAL_FEATURES`,
`MALAYSIA_CATEGORICAL`, `INDIA_CATEGORICAL`, `MALAYSIA` (0) / `INDIA` (1) cohort codes,
the per-type retention cut-offs (`BINARY_AUROC_THRESHOLD`, `BINARY_F1_THRESHOLD`,
`MULTICLASS_F1_THRESHOLD`, `MULTICLASS_AUROC_THRESHOLD`, `REGRESSION_R2_THRESHOLD`;
`ACCURACY_THRESHOLD` is the deprecated single-number form),
`DECISION_THRESHOLD` (0.5), `SCREENING_THRESHOLD` (0.10), `THRESHOLD_GRID` (0.10-0.90
in steps of 0.05), `MIN_POSITIVE_PREDICTIONS_FOR_PPV` (10), `ECE_BINS` (10),
`N_BOOTSTRAP` (2000), `ALPHA` (0.05), `CI_METHODS` (which interval method each
quantity uses), `DCA_CLINICAL_RANGE`, `MODEL_DISPLAY_NAMES`.
Functions: `set_seed(seed=SEED)`, `results_path(*parts) -> Path` (creates the directory).

## `sga.data`
- `scaling`: `scale_feature_train(df, method="std") -> (df, std_or_min, mean_or_max)`,
  `scale_feature_test(df, std_or_min, mean_or_max, method="std")`,
  `descale_feature(...)`, `scale_single_value(...)`
- `cleaning`: `CONTINUOUS_FEATURE_LOGICAL_RANGE`, `remove_illogical_values(df, keep_nan=True, return_indices=False)` (in place),
  `remove_duplicates(...)`, `fit_iterative_imputer(df, random_state=SEED, verbose=False)`,
  `apply_iterative_imputer(imputer, df)` (fills only the missing cells),
  `iterative_impute(df, ...)` (fit and apply in one step; use the pair above wherever a
  train / held-out split exists)
- `encoding`: `is_numeric`, `convert_feature_to_label`, `convert_feature_to_one_hot`, `one_hot`,
  `encode_india_categoricals`
- `centiles`: `merge_groundtruth(df, others=True, divide_by_1000=True, chart="i21")`,
  `compute_efw(row)`, `compute_efw_centile(df)`, `assign_sga_label(row)`
- `splits`: `assign_folds(df, num_folds, label, id_exist, seed, keep_id=True)`,
  `split_complete_and_addon(df, categorical_col, num_folds, label, id_exist,
  impute_data=False, seed, external_fold=4) -> (complete, add_on)` — `impute_data=False`
  leaves the gaps for the fold builder to fill
- `statistics`: `normality_and_parametric_test(df, df_sga, df_non_sga, feature, continuous, results)`,
  `get_count_categorical_feature`, `get_mean_continuous_feature`, `save_results_to_csv`, `save_results_to_excel`

## `sga.imputation`
- `registry`: `IMPUTATION_MODELS` (feature -> directory), `model_dir(feature)`
- `fold_imputers`: `fit_fold_imputers(source_frames, targets, seed=SEED, verbose=False) -> {feature: model}`.
  Refits the cross-domain imputers on ONE fold's training partition and returns them in
  memory. `source_frames` maps `"malaysia"`/`"india"` to that cohort's training frame;
  `targets` is a list of `(feature, kind)` pairs. Pass the result to
  `impute_df(..., imputers=...)` — this is the mechanism that keeps a fold's held-out rows
  out of the model generating their harmonized features. Hyperparameters are reused from
  `scripts/02_train_imputers.py` when available.
- `apply`: `impute_df(df, impute_features, binaryclass_features=(), multiclass_features=(), regression_features=(), verbose=False, imputers=None)`
  — `imputers` takes the mapping from `fit_fold_imputers`; features absent from it fall back
  to the persisted development-block weights, which is correct only outside cross-validation.
  `select_features_with_threshold(features, threshold=None, selection_type, keep_maternal_history=False, verbose=False) -> (selected, removed)`
  — applies the PER-TYPE retention criteria from the Methods (binary: AUROC > 0.80 and
  F1 > 0.50; multiclass: F1 > 0.80 and AUROC > 0.50; continuous: R² > 0.80). `threshold`
  overrides only the primary cut-off; the secondary safeguards always apply.
  `IMPUTER_INPUT_FEATURES`

## `sga.pipeline`
- `dataset`: `load_cohort(subdir, chart, base_dir=None, exclude_external_fold=True, external_fold=4) -> [df, add_on]`,
  `load_both_cohorts(...) -> (malaysia_pair, india_pair)`,
  `separate_df_and_df_add_on(dfs, label, id_exists, additional_drop_columns=None) -> (df, add_on, cats, conts, features)`,
  `process_raw_train_and_test_df(df, add_on, fold, id_exists, add_noise_features=None,
  seed=SEED, label=LABEL, impute_within=True, return_imputer=False) -> (train, test[, imputer])`
  — fits the within-feature imputer on the training rows of THIS fold,
  `impute_within_feature(train_df, test_df=None, label=LABEL, seed=SEED) -> (train, test, imputer)`,
  `apply_within_feature_imputer(imputer, df, label=LABEL)`,
  `scale_sample_train_and_test_df(train, test, validation, cats, conts, test_on_other_csv=None, sample_size=None, seed=SEED) -> (train, test, validation, std_or_min, mean_or_max)`,
  `scale_training_partition(train_df, scale_factor, label, random_state)`,
  `stratified_subsample(df, n, label, seed)`, `cast_common_types(df)`
- `harmonized_fold`: `prepare_fold(msia_ds, india_ds, fold, selected_features=(),
  train_source="both", subsample_n=None, subsample_seed=SEED, label=LABEL,
  fit_imputers_per_fold=True, imputer_seed=SEED, msia_ds_full=None, india_ds_full=None,
  external_test_fold=None) -> FoldData`. Passing `external_test_fold` (with the full
  cohorts) keeps `fold`'s training partition and swaps the test partition for the external
  fold, so one development fold-model can be scored there with its own imputers and scaler.
  `MALAYSIA` / `INDIA` are re-exported from `sga.config`.
  `FoldData` exposes `.train_X .train_Y .test_X .test_Y .country_arr .cluster_ids .cats
  .features .n_msia_test .n_india_test .n_train_raw` and also supports `fold["train_X"]`
  access.
- `external_fold`: `build_calibrated_external_fold(...) -> CalibratedExternalFold` — the one
  construction every externally-reported number uses: fits the four development fold-models,
  calibrates on their pooled held-out predictions, and scores the external fold with the
  fold-model that had the highest validation AUROC.
  `development_fold_models(...)`, `development_out_of_fold_predictions(...)`,
  `TARGET_SENSITIVITY`. `CalibratedExternalFold` exposes `.y_true .p_raw .p_calibrated
  .country_arr .cluster_ids .model .calibrator .youden_threshold .sensitivity_threshold
  .selected_fold .validation_auroc .development_prevalence .external_prevalence`,
  `.splits()`, `.composition()` and `.bootstrap_unit()`.
- `model_io`: the single implementation of weight loading, scoring and prediction
  persistence used by every "no-retrain" analysis.
  Constants `CATBOOST` (`"catboost"`), `SKLEARN` (`"ml"`; `"sklearn"` accepted as a
  synonym), `DNN` (`"dnn"`), `WEIGHTS_SUBDIR` (`"model_weights"`).
  `weights_exist(download_path, n_folds=N_FOLDS_CV, weights_subdir=WEIGHTS_SUBDIR) -> bool`,
  `load_trained_model(family, download_path, fold, n_features=None, dnn_config=None,
  model_size="large", weights_subdir=WEIGHTS_SUBDIR)` (returns None when no weight
  file exists; pass `weights_subdir=""` for flat `model_<fold>.pkl` layouts),
  `predict_proba(model, family, X) -> np.ndarray`,
  `predict_labels_and_proba(model, family, X, threshold=DECISION_THRESHOLD) -> (y_pred, y_prob)`,
  `save_predictions(out_path, feature_df, y_true, y_pred, y_prob, country_arr=None)`
  (writes the features plus `Actual`, `Prediction`, `predicted_probability` and,
  when `country_arr` is given, a `country` column).
- `train_unified`: `build_harmonized_folds`, `train_catboost_unified`,
  `train_sklearn_unified`, `train_dnn_unified` and `select_cross_domain_features`
  all accept `selected_features=None`. Passing an explicit list (an empty list
  counts) bypasses the imputation-quality gate and retains exactly those
  cross-domain features, marking every other candidate removed — the replacement
  for the round-1 monkey-patching of `select_features_with_threshold`.
  `build_harmonized_folds` additionally accepts `feature_subset=None` (restrict the yielded
  model matrix to these columns, for scoring a model trained on a narrower space — e.g. the
  common-feature country baselines), `fit_imputers_per_fold=True` and `imputer_seed=SEED`.
- `inference`: `run_external_inference(...)` accepts `selected_features=None` and
  `feature_subset=None`, so the `unified_common_*` arm of Figure 3 and the country baselines
  can be scored on the same held-out rows as the harmonized models.
- `inference`: `load_model`, `predict` and `weights_present` are aliases of the
  `model_io` functions above, kept for the `scripts/04*` entry points.

## `sga.evaluation`
- `metrics`: `EVAL_SPLITS`, `METRIC_COLUMNS`, `expected_calibration_error(y_true, y_prob, n_bins=10)`,
  `confusion_counts(y_true, y_pred) -> {tp, fp, fn, tn}`, `RATE_COUNTS`,
  `rate_numerator_denominator(counts, rate)`, `confusion_rates(y_true, y_pred, undefined=0.0)`,
  `basic_metrics(y_true, y_prob, threshold=0.5)`, `full_metrics(y_true, y_pred, y_prob)`,
  `youden_threshold`, `threshold_for_sensitivity` (returns the HIGHEST cut-off still
  reaching the target)
- `proportions`: closed-form score intervals for proportions and their differences —
  `wilson_ci(successes, n, alpha=ALPHA) -> Interval`,
  `newcombe_difference_ci(k1, n1, k2, n2, alpha=ALPHA) -> Interval`,
  `rate_with_ci(numerator, denominator)` and `difference_with_ci(...)` (return
  `NOT_ESTIMABLE` on an empty denominator rather than a misleading 0.0),
  `percentile_ci(values, alpha=ALPHA)`, `format_interval(...)`. `Interval` carries
  `.point .low .high`, `.is_estimable`, `.format()`, `.as_dict(prefix)` and
  `.excludes(value=0.0)`
- `calibration`: `logit`, `fit_platt`, `apply_platt`, `platt_cross_fitted(y_true, p_raw, n_splits=5, seed=SEED)`,
  `reliability_curve(y_true, y_prob, n_bins=10, min_count=1)`
- `delong`: `compute_midrank`, `delong_test(y, p1, p2) -> (auc1, auc2, z, p)`,
  `holm_bonferroni_correction(p_values)` (alias `holm_correction`),
  `pairwise_delong(y_true, {name: probs}, correct=True) -> DataFrame` — adjusted values in
  `p_value_holm_bonferroni`, uncorrected in `p_value`
- `bootstrap`: `bootstrap_ci(values, ...)`, `bootstrap_metric_ci(y_true, y_prob, metric, ..., cluster_ids=None)`,
  `bootstrap_statistic_ci(y_true, y_prob, statistic, ..., cluster_ids=None)` for any callable
  statistic (used for the threshold-dependent sweep behind appendix Table S4),
  `bootstrap_difference_ci(y_true, p_a, p_b, metric, ...)`,
  `bootstrap_rate_grid(y_true, y_prob, thresholds, rates=("sensitivity", "specificity"), ...)`
  — one shared set of resamples across a whole threshold grid.
  Scope: the bootstrap is for AUROC, AUPRC, ECE and Brier; proportions use
  `sga.evaluation.proportions`
- `fairness`: `fairness_at_threshold(y_true, y_prob, country_arr, threshold, ...)` — every
  cohort-level quantity at one cut-off, with Wilson intervals on the rates and Newcombe
  intervals on the signed Malaysia-minus-India differences;
  `fairness_threshold_sweep(...)` across `THRESHOLD_GRID`;
  `fairness_points(...)` (point estimates only, for resampling);
  `bootstrap_fairness_sweep(..., cluster_ids=None, stratify_by_cohort=True)`;
  `cohort_counts`, `cohort_rates`, `GROUPS`, `DIFFERENCE_NAMES`;
  `compute_group_rates` and `compute_fairness_metrics` (absolute gaps, no uncertainty)
- `country`: `evaluate_splits(y_true, y_pred, y_prob, country_arr) -> {"total"|"malaysia"|"india": metrics}`,
  `append_fold_rows(all_results, split_metrics, model_name, fold, extra=None)`, `summarize(df, group_cols)`,
  `save_country_results(all_results, save_dir, prefix="country", model_name=None, group_cols=...)`
- `dca`: `decision_curve_analysis(y_true, y_prob, thresholds=None) -> (thresholds, net_benefits, treat_all)`,
  `plot_dca(dca_results, scenario_name, save_dir, show_defaults=True, zoom=False)`,
  `save_dca_csv(dca_results, scenario_name, save_dir)`

## `sga.models`
- `architecture`: `FNNClassifierTri3`, `FNNClassifierTri3_Medium`, `FNNClassifierTri3_Small`,
  `FNNClassifierTri3_Test`, `FNNClassifierTri3_Calibration`, `MODEL_SIZES`
- `hyperparameters`: `rf_hyperparameters`, `lr_hyperparameters`, `svc_hyperparameters`,
  `stacking_hyperparameters`, `catboost_hyperparameters`, `multiclass_catboost_hyperparameters`,
  `regression_catboost_hyperparameters`, `regression_impute_catboost_hyperparameters`,
  `improved_regression_catboost_hyperparameters`, and the regression/multiclass ML grids
- `loops`: `train_dnn`, `validate_dnn`, `test_dnn`
- `torch_utils`: `DEVICE`, `convert_to_tensor_dataloader`, `adjust_learning_rate`,
  `compute_permutation_importance`
- `ensemble`: `build_base_estimators()`, `build_stacking_classifier()`
- `estimators`: `train_lr(train_X, train_Y, seed=SEED)` (grid-searched Logistic
  Regression used by the analysis scripts), `LR_GRID_SEARCH_CV_FOLDS`, `LR_MAX_ITER`
- `domain_adaptation`: the CORAL / IRM / DANN mechanisms used as comparators in
  manuscript Table 5. `coral_align(train_X, train_y, domain_train, source=0,
  target=1, ridge=CORAL_RIDGE) -> (aligned_X, aligned_y, transform)` (whitens the
  source cohort and recolours it onto the target covariance),
  `irm_penalty(logits, y, device) -> tensor` (IRM-v1 squared dummy-scale
  gradient), `GradientReversalFunction`, `GradientReversalLayer(lambda_=1.0)`
  (assign to `.lambda_` to ramp the adversarial strength),
  `DomainClassifier(input_size, hidden_size=8)`, constant `CORAL_RIDGE`.
  Training schedules and hyperparameters stay with the reporting experiment.

## `sga.reporting`
- `metrics_tables`: `calculate_metrics(cm)`, `calc_metrics_with_ci`, `calc_metrics_updated`,
  `display_metrics`, `display_metrics_updated`, `display_ml_metrics`,
  `display_ml_regression_metrics`, `display_metrics_for_multiple_iterations`
- `plots`: `display_roc_curve`, `display_roc_curve_binary`, `display_cm`, `plot_confusion_matrix`,
  `display_training_loss`, `display_validation_loss`, `display_trees`
- `artifacts`: `save_train_test_set`, `save_output_layer_dict`
- `importance`: `compute_and_plot_permutation_importance`, `rf_f_importances`, `lr_f_importances`,
  `svc_f_importances`, `catboost_f_importances`, `save_permutation_importance_list`
- `shap_analysis`: `perform_SHAP`, `display_shap`, `new_shap_dict`, `load_shap_summaries`
- `figures`: the manuscript result figures. `MODEL_ORDER`, `STRATEGY_COLORS`,
  `COHORT_LABELS`, `apply_manuscript_style()`;
  `plot_auroc_comparison(...)` (Figure 3),
  `plot_auroc_by_gestational_week(...)` / `plot_auroc_by_maternal_age(...)` (Figure 5),
  and for Figure 7 `load_size_summary(csv_path, metric, max_size)`,
  `plot_figure7_combined/…_auroc_only/…_percentage_only/…_by_cohort(...)`,
  `figure7_plotted_values(...)`, `warn_if_figure7_clipped(...)`, with colour and marker
  fixed per model in `FIGURE7_MODELS`;
  `plot_threshold_sweep(sweep, save_path)` and `plot_fairness_vs_threshold(sweep, save_path)`
