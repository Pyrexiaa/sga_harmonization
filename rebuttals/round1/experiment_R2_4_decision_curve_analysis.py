"""Standalone decision-curve analysis.

Runs the unified CatBoost pipeline end to end over the four cross-validation
folds, pools the held-out predicted probabilities, and computes the net benefit

    NB(pt) = TP/n - (FP/n) * pt / (1 - pt)

across threshold probabilities 0.01 to 0.99, against the treat-all and
treat-none default strategies. This is the self-contained version of the DCA;
``experiment_R2_3_additional_metrics.py`` produces the multi-model panel that
becomes manuscript Figure 4.

Prerequisites:
    Run experiment_R0_baseline_retrain.py first (the pretrained CatBoost
    imputers).

Run:
    python -m rebuttals.round1.experiment_R2_4_decision_curve_analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV

from sga.config import ACCURACY_THRESHOLD, N_FOLDS_CV, ROUND1_DIR, SEED, set_seed
from sga.evaluation.dca import decision_curve_analysis, plot_dca, save_dca_csv
from sga.models.hyperparameters import catboost_hyperparameters
from sga.pipeline.dataset import load_both_cohorts
from sga.pipeline.train_unified import build_harmonized_folds

SAVE_DIR = ROUND1_DIR / "R2_4_decision_curve_analysis"

MODEL_NAME = "CatBoost (domain-generalized)"
SCENARIO_NAME = "Domain-Generalized CatBoost"


def collect_out_of_fold_probabilities(msia_ds, india_ds, num_of_folds=N_FOLDS_CV,
                                      accuracy_threshold=ACCURACY_THRESHOLD,
                                      smoting=True):
    """Train the unified CatBoost per fold and pool its held-out probabilities."""
    all_y_true, all_y_prob = [], []

    for prepared in build_harmonized_folds(
        msia_ds,
        india_ds,
        num_of_folds=num_of_folds,
        accuracy_threshold=accuracy_threshold,
        smoting=smoting,
    ):
        train_X = prepared.train_X
        train_Y = prepared.train_Y.astype(int)
        test_X = prepared.test_X
        test_Y = prepared.test_Y.astype(int)

        search = GridSearchCV(
            estimator=CatBoostClassifier(class_names=[0, 1]),
            param_grid=catboost_hyperparameters(),
            scoring=make_scorer(roc_auc_score), cv=5,
        )
        search.fit(train_X, train_Y, eval_set=(test_X, test_Y), verbose=0)

        model = CatBoostClassifier(**search.best_params_, class_names=[0, 1])
        model.fit(
            Pool(data=train_X, label=train_Y,
                 cat_features=prepared.categorical_features),
            eval_set=Pool(data=test_X, label=test_Y,
                          cat_features=prepared.categorical_features),
            verbose=0,
        )

        y_prob = np.asarray(model.predict_proba(test_X))[:, 1]
        y_true = np.asarray(test_Y)
        print(f"    Fold {prepared.fold}: AUC={roc_auc_score(y_true, y_prob):.4f}, "
              f"n_test={len(y_true)}")
        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    print(f"    Overall pooled AUC: {roc_auc_score(all_y_true, all_y_prob):.4f}")
    return all_y_true, all_y_prob


def run_experiment():
    """Run the pipeline, compute the decision curve and save the artefacts."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    msia_ds, india_ds = load_both_cohorts()
    print(f"  Running the unified CatBoost pipeline ({N_FOLDS_CV}-fold CV)")
    print(f"  accuracy_threshold={ACCURACY_THRESHOLD}, SMOTE=True")

    y_true, y_prob = collect_out_of_fold_probabilities(msia_ds, india_ds)

    dca_results = {MODEL_NAME: decision_curve_analysis(y_true, y_prob)}
    plot_dca(dca_results, SCENARIO_NAME, str(SAVE_DIR))
    save_dca_csv(dca_results, SCENARIO_NAME, str(SAVE_DIR))

    overall_auc = roc_auc_score(y_true, y_prob)
    auc_path = SAVE_DIR / "auc_summary.csv"
    pd.DataFrame([{
        "model": MODEL_NAME,
        "auc": overall_auc,
        "n_samples": len(y_true),
        "prevalence": y_true.mean(),
    }]).to_csv(auc_path, index=False)
    print(f"  [DCA] AUC summary saved: {auc_path}")

    predictions_path = SAVE_DIR / "predictions.csv"
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(
        predictions_path, index=False)
    print(f"  [DCA] Predictions saved: {predictions_path}")

    print("\nR2-4: DECISION CURVE ANALYSIS COMPLETE")
    print(f"    Results saved to: {SAVE_DIR}")
    print(f"    Overall AUC: {overall_auc:.4f}")
    print(f"    Total samples: {len(y_true)}")
    print(f"    SGA prevalence: {y_true.mean():.3f}")
    return y_true, y_prob


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
