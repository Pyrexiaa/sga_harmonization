"""Training versus testing partitions and source versus target cohorts.

Appendix Table S3 compares the ten common predictors between the training and
testing partitions of each cohort; the testing partition is fold 4 of the complete
records, and the training partition is every other record. Appendix Table S5 gives
the standardised mean difference (SMD) and overlap coefficient (OVL) of the same
predictors between the source cohort (Malaysia) and the target cohort (India).

Run:
    python -m rebuttals.round3.experiment_R3_6_cohort_shift_tables
"""

from __future__ import annotations

import pandas as pd

from sga.config import (
    COMMON_FEATURES,
    EXTERNAL_TEST_FOLD,
    INDIA_SUBDIR,
    MALAYSIA_SUBDIR,
    ROUND3_DIR,
)
from sga.data.statistics import covariate_overlap, partition_summary
from sga.pipeline.dataset import load_cohort

SAVE_DIR = ROUND3_DIR / "R3_6_cohort_shift_tables"
CONTINUOUS = [f for f in COMMON_FEATURES if f != "gender"]


def with_fetal_sex(frame):
    """Add readable fetal-sex columns (``gender`` is coded 1 = male, 0 = female)."""
    frame = frame.copy()
    frame["fetal_sex"] = frame["gender"].map({0: "Female", 1: "Male"})
    frame["female"] = (frame["gender"] == 0).astype(float).where(frame["gender"].notna())
    return frame


def partitions(subdir):
    """``(training, testing)`` frames of one cohort, before any imputation."""
    complete, add_on = load_cohort(subdir, exclude_external_fold=False)
    testing = complete[complete["fold"] == EXTERNAL_TEST_FOLD]
    training = pd.concat([complete[complete["fold"] != EXTERNAL_TEST_FOLD], add_on])
    if "id" in testing.columns:
        training = training[~training["id"].isin(set(testing["id"]))]
    return with_fetal_sex(training), with_fetal_sex(testing)


def run_experiment():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    malaysia_train, malaysia_test = partitions(MALAYSIA_SUBDIR)
    india_train, india_test = partitions(INDIA_SUBDIR)
    table_s3 = partition_summary(
        {
            "Malaysia training": malaysia_train,
            "Malaysia testing": malaysia_test,
            "India training": india_train,
            "India testing": india_test,
        },
        CONTINUOUS,
        categorical=["fetal_sex"],
    )
    malaysia = with_fetal_sex(pd.concat(load_cohort(MALAYSIA_SUBDIR, exclude_external_fold=False)))
    india = with_fetal_sex(pd.concat(load_cohort(INDIA_SUBDIR, exclude_external_fold=False)))
    table_s5 = covariate_overlap(malaysia, india, CONTINUOUS, binary=["female"])

    table_s3.to_csv(SAVE_DIR / "tableS3_training_vs_testing.csv", index=False)
    table_s5.to_csv(SAVE_DIR / "tableS5_source_target_overlap.csv", index=False)
    print("Table S3\n" + table_s3.to_string(index=False))
    print("\nTable S5\n" + table_s5.round(2).to_string(index=False))
    print(f"\nSaved to: {SAVE_DIR}")
    return table_s3, table_s5


if __name__ == "__main__":
    run_experiment()
