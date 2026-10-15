"""Performance on records outside the held-out test partition (appendix Table S10).

The held-out test partition contains complete records only. Here the complete and
incomplete records of both cohorts are pooled, folds are reassigned over that union
by stratified group k-fold keyed on the pregnancy identifier, and each fold is held
out in turn while the model and every data-dependent step are fitted on the other
four (``sga.pipeline.incomplete_records``), so every record is scored once by a
model that never saw it.

Outputs:
    tableS10a_by_record_type.csv       complete vs incomplete records, per cohort
    tableS10b_by_imputed_predictors.csv  by how many common predictors were imputed
    tableS10c_unpaired_delong.csv      complete vs incomplete, unpaired DeLong
    record_type_by_imputed_predictors.csv  how the two stratifications overlap

Run:
    python -m rebuttals.round3.experiment_R3_1_incomplete_records
"""

from __future__ import annotations

import warnings

from sga.config import INDIA, INDIA_SUBDIR, MALAYSIA, MALAYSIA_SUBDIR, ROUND3_DIR, SEED, set_seed
from sga.pipeline.dataset import load_cohort
from sga.pipeline.incomplete_records import (
    build_union,
    delong_complete_vs_incomplete,
    record_type_by_burden,
    score_all_records,
    table_by_imputation_burden,
    table_by_record_type,
)

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND3_DIR / "R3_1_incomplete_records"


def run_experiment():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    malaysia = build_union(load_cohort(MALAYSIA_SUBDIR, exclude_external_fold=False), MALAYSIA, True)
    india = build_union(load_cohort(INDIA_SUBDIR, exclude_external_fold=False), INDIA, False)
    print(f"Malaysia: {len(malaysia)} scans, {malaysia['id'].nunique()} pregnancies")
    print(f"India: {len(india)} records")

    predictions = score_all_records(malaysia, india)
    predictions.to_csv(SAVE_DIR / "out_of_fold_predictions.csv", index=False)

    tables = {
        "tableS10a_by_record_type.csv": table_by_record_type(predictions),
        "tableS10b_by_imputed_predictors.csv": table_by_imputation_burden(predictions),
        "tableS10c_unpaired_delong.csv": delong_complete_vs_incomplete(predictions),
    }
    for name, table in tables.items():
        table.to_csv(SAVE_DIR / name, index=False)
        print(f"\n{name}\n{table.to_string(index=False)}")
    record_type_by_burden(predictions).to_csv(SAVE_DIR / "record_type_by_imputed_predictors.csv")
    print(f"\nSaved to: {SAVE_DIR}")
    return tables


if __name__ == "__main__":
    run_experiment()
