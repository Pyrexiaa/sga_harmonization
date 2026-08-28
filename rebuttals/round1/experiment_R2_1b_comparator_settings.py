"""Comparator implementation details (appendix Table S6).

Appendix Table S6 states the preprocessing, architecture, source/target setup,
hyperparameters and seed of the four domain-generalization comparators. Those
values live as module constants in
``experiment_R2_1_domain_generalization_comparison``; this script reads them from
that module and writes the table, so the appendix cannot silently drift away from
the code that produced Table 5 of the main text.

No model is trained and no data is read -- it is a settings dump. Run it after
changing any comparator hyperparameter and paste the CSV into the appendix.

Run:
    python -m rebuttals.round1.experiment_R2_1b_comparator_settings
"""

from __future__ import annotations

import pandas as pd

from sga.config import COMMON_FEATURES, ROUND1_DIR, SEED

from rebuttals.round1.experiment_R2_1_domain_generalization_comparison import (
    DANN_LAMBDA,
    FNN_BATCH_SIZE,
    FNN_DROPOUT,
    FNN_EPOCHS,
    FNN_LAYER_SIZE,
    FNN_LR,
    FNN_WEIGHT_DECAY,
    IRM_ANNEAL_STEPS,
    IRM_LAMBDA,
)

SAVE_DIR = ROUND1_DIR / "R2_1b_comparator_settings"

#: Logistic Regression settings used by ``train_linear`` in the R2-1 experiment.
LR_MAX_ITER = 1000
LR_PENALTY = "L2 (scikit-learn default)"
LR_CLASS_WEIGHT = "balanced"


def _fnn_settings():
    """The shared feed-forward backbone, as the appendix phrases it."""
    return (
        f"Adam, lr {FNN_LR:g}, weight decay {FNN_WEIGHT_DECAY:g}, BCE loss, "
        f"{FNN_EPOCHS} epochs, batch {FNN_BATCH_SIZE} "
        f"(dropout {FNN_DROPOUT:g}, layer size {FNN_LAYER_SIZE})"
    )


def build_table():
    """Assemble Table S6 from the constants the comparators actually use."""
    common = f"Common features ({len(COMMON_FEATURES)}): {', '.join(COMMON_FEATURES)}"
    fnn = _fnn_settings()

    return pd.DataFrame(
        {
            "": [
                "Preprocessing",
                "Architecture",
                "Source & Target",
                "Hyperparameters",
                "Seed",
                "Feature set",
            ],
            "CORAL+LR": [
                "Common features with covariance alignment",
                "Logistic Regression",
                "Source = Malaysia / Target = India / "
                "Alignment fit on source to target covariances",
                f"max_iter={LR_MAX_ITER}, {LR_PENALTY}, class_weight=\"{LR_CLASS_WEIGHT}\"",
                SEED,
                common,
            ],
            "CORAL+FNN": [
                "Common features with covariance alignment",
                "Feed Forward Neural Network",
                "Source = Malaysia / Target = India / "
                "Alignment fit on source to target covariances",
                f"FNN settings ({fnn})",
                SEED,
                common,
            ],
            "IRM+FNN": [
                "Common features with standard scaling",
                "Feed Forward Neural Network",
                "Two separate environments (Malaysia, India)",
                f"FNN settings + IRM penalty lambda = {IRM_LAMBDA:g}, "
                f"switched on after epoch {IRM_ANNEAL_STEPS} of {FNN_EPOCHS} "
                f"(penalty weight 1.0 before that)",
                SEED,
                common,
            ],
            "DANN+FNN": [
                "Common features with standard scaling",
                "Feed Forward Neural Network",
                "Source = Malaysia / Target = India / "
                "Domain label as adversarial target",
                f"FNN settings + gradient-reversal weight ramped 0 -> {DANN_LAMBDA:g} "
                f"over training via 2/(1+exp(-10p))-1, p = epoch/{FNN_EPOCHS}",
                SEED,
                common,
            ],
        }
    )


def run_experiment():
    """Write Table S6 and print it."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    path = SAVE_DIR / "tableS6_comparator_settings.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print(
        "\nNote: the IRM penalty is applied as a hard switch at the epoch boundary "
        "and the DANN weight follows the Ganin ramp rather than being held fixed. "
        "Appendix Table S6 should describe them the same way."
    )
    print(f"\nSaved: {path}")
    return table


if __name__ == "__main__":
    run_experiment()
