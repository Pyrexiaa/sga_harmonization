"""Diagnostic for the amniotic fluid condition (af) imputer (appendix Table S2c).

Rebuilds the af imputation task exactly as ``scripts/02_train_imputers.py`` trains it
(folds 0-2 train, fold 3 validates, fold 4 untouched), scores the persisted af
imputer on the validation fold and reports:

(a) the class distribution of af in the training, validation and test folds;
(b) the confusion matrix on the validation fold with per-class recall;
(c) weighted F1 and the macro one-vs-rest AUROC with its 95% CI, against an
    always-predict-majority baseline - the two extra multiclass criteria;
(d) the af distribution the imputer writes into the Indian cohort.

Run:
    python -m rebuttals.round3.experiment_R2_2_af_diagnostic
"""

from __future__ import annotations

import warnings


from sga.config import INDIA_SUBDIR, MALAYSIA_SUBDIR, ROUND3_DIR, SEED, set_seed
from sga.data.cleaning import remove_illogical_values
from sga.imputation.apply import IMPUTER_INPUT_FEATURES, load_persisted_imputer, impute_df
from sga.imputation.diagnostics import class_distribution, multiclass_quality
from sga.imputation.train_imputers import (
    drop_rare_classes,
    handle_missing_values,
    prepare_frames,
    split_train_validation_test,
)
from sga.pipeline.dataset import impute_within_feature, load_cohort

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND3_DIR / "R2_2_af_diagnostic"
TARGET = "af"
AF_LEVELS = {0: "oligohydramnios", 1: "normal", 2: "polyhydramnios"}


def af_task_frames():
    """Training, validation and test frames of the af imputation task."""
    malaysia = load_cohort(MALAYSIA_SUBDIR, exclude_external_fold=False)
    df, add_on, categorical, _, features = prepare_frames(malaysia, TARGET, id_exists=True)
    frames = split_train_validation_test(df, add_on, id_exists=True)
    train, validation, test = handle_missing_values(frames, features, categorical, TARGET, seed=SEED)
    train, rare = drop_rare_classes(train, TARGET)
    validation = validation[~validation[TARGET].isin(rare)].reset_index(drop=True)
    test = test[~test[TARGET].isin(rare)].reset_index(drop=True)
    for frame in (train, validation, test):
        remove_illogical_values(frame)
    return train, validation, test, features


def india_imputed_distribution(model):
    """What the af imputer writes into the Indian complete and incomplete records."""
    complete, add_on = load_cohort(INDIA_SUBDIR, exclude_external_fold=False)
    inputs = list(IMPUTER_INPUT_FEATURES)
    _, add_on_inputs, _ = impute_within_feature(complete[inputs], add_on[inputs], label=None, seed=SEED)
    imputed = {
        "complete records": impute_df(complete[inputs], [TARGET], multiclass_features=[TARGET],
                                      imputers={TARGET: model}),
        "incomplete records": impute_df(add_on_inputs, [TARGET], multiclass_features=[TARGET],
                                        imputers={TARGET: model}),
    }
    return class_distribution(imputed, TARGET)


def run_experiment():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    train, validation, test, features = af_task_frames()
    model = load_persisted_imputer(TARGET, "multiclass")

    distribution = class_distribution(
        {"training fold": train, "validation fold": validation, "test fold": test}, TARGET
    )
    quality = multiclass_quality(
        validation[TARGET],
        model.predict(validation[features]),
        model.predict_proba(validation[features]),
        model.classes_,
        train_y=train[TARGET],
    )
    india = india_imputed_distribution(model)

    distribution.to_csv(SAVE_DIR / "tableS2c_a_class_distribution.csv", index=False)
    quality.confusion_frame().to_csv(SAVE_DIR / "tableS2c_b_confusion_matrix.csv")
    quality.summary_rows().to_csv(SAVE_DIR / "tableS2c_c_summary.csv", index=False)
    india.to_csv(SAVE_DIR / "tableS2c_d_india_imputed_distribution.csv", index=False)

    print(f"af levels: {AF_LEVELS}")
    print("\n(a) class distribution\n" + distribution.to_string(index=False))
    print("\n(b) validation-fold confusion matrix\n" + quality.confusion_frame().to_string())
    print("\n(c) against the always-predict-majority baseline\n"
          + quality.summary_rows().to_string(index=False))
    print("\n(d) imputed af in the Indian cohort\n" + india.to_string(index=False))
    print(f"\nRetained under the multiclass rule: {quality.retained}")
    print(f"Saved to: {SAVE_DIR}")
    return quality


if __name__ == "__main__":
    run_experiment()
