"""CORAL / IRM / DANN comparators (manuscript Table 5).

run with Logistic Regression so the comparison spans a neural and a linear
classifier. IRM and DANN are neural-only.

    ARM A (ten common features)  Naive FNN, Naive LR, CORAL+FNN, CORAL+LR,
                                 IRM+FNN, DANN+FNN
    ARM B (proposed)             Harmonized FNN, Harmonized LR

Each arm is scored total / Malaysia / India; trained weights are saved per
method and fold and reused on re-runs (no retraining when they exist).

Prerequisites:
    Run experiment_R0_baseline_retrain.py first (the pretrained CatBoost
    imputers used by ARM B).

Run:
    python -m rebuttals.round1.experiment_R2_1_domain_generalization_comparison
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from sga.config import (
    ACCURACY_THRESHOLD,
    LABEL,
    N_FOLDS_CV,
    ROUND1_DIR,
    SEED,
    set_seed,
)
from sga.evaluation.country import evaluate_splits, summarize
from sga.evaluation.metrics import EVAL_SPLITS, full_metrics
from sga.models.architecture import FNNClassifierTri3
from sga.models.domain_adaptation import (
    DomainClassifier,
    GradientReversalLayer,
    coral_align,
    irm_penalty,
)
from sga.models.torch_utils import DEVICE
from sga.pipeline.dataset import (
    load_both_cohorts,
    process_raw_train_and_test_df,
    separate_df_and_df_add_on,
)
from sga.pipeline.harmonized_fold import INDIA, MALAYSIA
from sga.pipeline.train_unified import build_harmonized_folds

warnings.filterwarnings("ignore")

SAVE_DIR = ROUND1_DIR / "R2_1_domain_generalization"
WEIGHTS_DIR = SAVE_DIR / "weights"

# Shared backbone hyperparameters (the paper's unified DNN configuration).
FNN_DROPOUT = 0.20
FNN_LAYER_SIZE = 4
FNN_EPOCHS = 100
FNN_BATCH_SIZE = 256
FNN_LR = 1e-3
FNN_WEIGHT_DECAY = 5e-3

IRM_LAMBDA = 1e4
IRM_ANNEAL_STEPS = 50  # warm up with plain ERM before the penalty kicks in
DANN_LAMBDA = 1.0

# Display name -> weights sub-directory.
METHOD_KEYS = {
    "Naive_FNN (10 common, ERM)": "naive_fnn",
    "Naive_LR (10 common, ERM)": "naive_lr",
    "CORAL_FNN (10 common)": "coral_fnn",
    "CORAL_LR (10 common)": "coral_lr",
    "IRM_FNN (10 common)": "irm_fnn",
    "DANN_FNN (10 common)": "dann_fnn",
    "Harmonized_FNN (proposed)": "harmonized_fnn",
    "Harmonized_LR (proposed)": "harmonized_lr",
}


def _save_weights(method_name, fold, save_dict):
    """Persist a trained model: torch backbones as ``.pt``, sklearn as ``.pkl``."""
    directory = WEIGHTS_DIR / METHOD_KEYS[method_name]
    directory.mkdir(parents=True, exist_ok=True)
    if save_dict.get("framework") == "sklearn":
        path = directory / f"model_{fold}.pkl"
        with open(path, "wb") as handle:
            pickle.dump(save_dict["model"], handle)
    else:
        path = directory / f"model_{fold}.pt"
        torch.save(save_dict, path)
    return str(path)


def _load_and_predict(method_name, fold, test_X, input_size):
    """Return ``(y_pred, y_prob)`` from saved weights, or None if absent."""
    directory = WEIGHTS_DIR / METHOD_KEYS[method_name]
    torch_path = directory / f"model_{fold}.pt"
    pickle_path = directory / f"model_{fold}.pkl"

    if torch_path.exists():
        checkpoint = torch.load(torch_path, map_location=DEVICE)
        net = FNNClassifierTri3(
            checkpoint.get("input_size", input_size),
            checkpoint.get("dropout", FNN_DROPOUT),
            checkpoint.get("layer_size", FNN_LAYER_SIZE),
        )
        net.load_state_dict(checkpoint["state_dict"])
        net.to(DEVICE)
        net.eval()
        with torch.no_grad():
            logits = (
                net(torch.as_tensor(np.asarray(test_X), dtype=torch.float32, device=DEVICE))
                .squeeze(-1).cpu().numpy().reshape(-1)
            )
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return (probabilities >= 0.5).astype(int), probabilities

    if pickle_path.exists():
        with open(pickle_path, "rb") as handle:
            model = pickle.load(handle)
        probabilities = model.predict_proba(test_X)[:, 1]
        return (probabilities >= 0.5).astype(int), probabilities

    return None


def _record_splits(all_results, method_name, fold, y_true, y_pred, y_prob, country_arr):
    """Append one metric row per evaluation split for one method and fold."""
    if country_arr is not None and len(country_arr) == len(y_true):
        split_metrics = evaluate_splits(y_true, y_pred, y_prob, country_arr)
    else:
        if country_arr is not None:
            print(f"    [warn] {method_name}: country array ({len(country_arr)}) "
                  f"!= test rows ({len(y_true)}); reporting TOTAL only.")
        split_metrics = {"total": full_metrics(y_true, y_pred, y_prob)}

    for split in EVAL_SPLITS:
        metrics = split_metrics.get(split)
        if not metrics:
            continue
        row = {"method": method_name, "fold": fold, "eval_split": split}
        row.update(metrics)
        all_results.append(row)
    return split_metrics


def _fnn_save_dict(net, input_size, extra=None):
    """Checkpoint dictionary for one trained backbone."""
    checkpoint = {
        "framework": "torch",
        "state_dict": net.state_dict(),
        "input_size": input_size,
        "dropout": FNN_DROPOUT,
        "layer_size": FNN_LAYER_SIZE,
        "arch": "FNNClassifierTri3",
    }
    if extra:
        checkpoint.update(extra)
    return checkpoint


def _score(net, test_X, test_y):
    """Evaluate a trained backbone on a test matrix."""
    net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(test_X, dtype=torch.float32).to(DEVICE)).squeeze(-1)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return test_y, (probabilities >= 0.5).astype(int), probabilities


def train_fnn(train_X, train_y, test_X, test_y, input_size, save_extra=None):
    """Train the shared backbone with plain ERM (the Naive / Harmonized arms)."""
    net = FNNClassifierTri3(input_size, FNN_DROPOUT, FNN_LAYER_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=FNN_LR, weight_decay=FNN_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(train_X, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        ),
        batch_size=FNN_BATCH_SIZE, shuffle=True, drop_last=False,
    )

    net.train()
    for _ in range(1, FNN_EPOCHS + 1):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            loss = loss_fn(net(batch_X).squeeze(-1), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    y_true, y_pred, y_prob = _score(net, test_X, test_y)
    return y_true, y_pred, y_prob, _fnn_save_dict(net, input_size, save_extra)


def train_linear(train_X, train_y, test_X, test_y):
    """Train the linear ERM comparator (Logistic Regression, no grid search)."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    model.fit(train_X, train_y)
    probabilities = model.predict_proba(test_X)[:, 1]
    return (
        test_y,
        (probabilities >= 0.5).astype(int),
        probabilities,
        {"framework": "sklearn", "model": model, "arch": "LogisticRegression"},
    )


def train_coral_fnn(train_X, train_y, domain_train, test_X, test_y, input_size):
    """CORAL alignment followed by the shared backbone."""
    aligned_X, aligned_y, transform = coral_align(train_X, train_y, domain_train)
    return train_fnn(aligned_X, aligned_y, test_X, test_y, input_size,
                     save_extra={"coral_transform": transform})


def train_coral_linear(train_X, train_y, domain_train, test_X, test_y):
    """CORAL alignment followed by Logistic Regression."""
    aligned_X, aligned_y, transform = coral_align(train_X, train_y, domain_train)
    y_true, y_pred, y_prob, save_dict = train_linear(aligned_X, aligned_y, test_X, test_y)
    save_dict["coral_transform"] = transform
    return y_true, y_pred, y_prob, save_dict


def train_irm_fnn(train_X, train_y, domain_train, test_X, test_y, input_size):
    """Invariant Risk Minimization over the two cohorts as environments."""
    net = FNNClassifierTri3(input_size, FNN_DROPOUT, FNN_LAYER_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=FNN_LR, weight_decay=FNN_WEIGHT_DECAY)

    domain_train = np.asarray(domain_train)
    loaders = []
    for code in (MALAYSIA, INDIA):
        mask = domain_train == code
        loaders.append(
            DataLoader(
                TensorDataset(
                    torch.tensor(train_X[mask], dtype=torch.float32),
                    torch.tensor(train_y[mask], dtype=torch.long),
                ),
                batch_size=FNN_BATCH_SIZE, shuffle=True,
            )
        )
    msia_loader, india_loader = loaders

    net.train()
    for epoch in range(1, FNN_EPOCHS + 1):
        penalty_weight = 1.0 if epoch < IRM_ANNEAL_STEPS else IRM_LAMBDA
        india_iterator = iter(india_loader)
        for msia_X, msia_y in msia_loader:
            msia_X, msia_y = msia_X.to(DEVICE), msia_y.to(DEVICE)
            try:
                india_X, india_y = next(india_iterator)
            except StopIteration:
                india_iterator = iter(india_loader)
                india_X, india_y = next(india_iterator)
            india_X, india_y = india_X.to(DEVICE), india_y.to(DEVICE)

            msia_logits = net(msia_X)
            msia_loss = F.binary_cross_entropy_with_logits(
                msia_logits, msia_y.float().unsqueeze(1))
            india_logits = net(india_X)
            india_loss = F.binary_cross_entropy_with_logits(
                india_logits, india_y.float().unsqueeze(1))

            loss = (msia_loss + india_loss) / 2.0 + penalty_weight * (
                irm_penalty(msia_logits, msia_y, DEVICE)
                + irm_penalty(india_logits, india_y, DEVICE)
            ) / 2.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    y_true, y_pred, y_prob = _score(net, test_X, test_y)
    return y_true, y_pred, y_prob, _fnn_save_dict(net, input_size)


def train_dann_fnn(train_X, train_y, domain_train, test_X, test_y, input_size):
    """Domain-adversarial training through a gradient-reversal layer."""
    net = FNNClassifierTri3(input_size, FNN_DROPOUT, FNN_LAYER_SIZE).to(DEVICE)
    reversal = GradientReversalLayer(lambda_=DANN_LAMBDA)
    domain_head = DomainClassifier(FNN_LAYER_SIZE, hidden_size=max(4, FNN_LAYER_SIZE))
    domain_head.to(DEVICE)

    optimizer = torch.optim.Adam(
        list(net.parameters()) + list(domain_head.parameters()),
        lr=FNN_LR, weight_decay=FNN_WEIGHT_DECAY,
    )
    task_loss_fn = nn.BCEWithLogitsLoss()
    domain_loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(train_X, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
            torch.tensor(np.asarray(domain_train), dtype=torch.float32),
        ),
        batch_size=FNN_BATCH_SIZE, shuffle=True, drop_last=False,
    )

    net.train()
    domain_head.train()
    for epoch in range(1, FNN_EPOCHS + 1):
        progress = epoch / FNN_EPOCHS
        reversal.lambda_ = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
        for batch_X, batch_y, batch_domain in loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_domain = batch_domain.to(DEVICE)
            representation = net.get_x_after_third_layer(batch_X)
            task_loss = task_loss_fn(net.layer4(representation).squeeze(-1), batch_y)
            domain_logits = domain_head(reversal(representation)).squeeze(-1)
            loss = task_loss + domain_loss_fn(domain_logits, batch_domain)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    y_true, y_pred, y_prob = _score(net, test_X, test_y)
    save_dict = _fnn_save_dict(net, input_size, extra={
        "domain_clf_state_dict": domain_head.state_dict(),
        "domain_clf_input_size": FNN_LAYER_SIZE,
    })
    return y_true, y_pred, y_prob, save_dict


def _evaluate_method(all_results, method_name, fold, test_X, test_y, input_size,
                     country_arr, trainer):
    """Reload saved weights when present, otherwise train and save; then record."""
    loaded = _load_and_predict(method_name, fold, test_X, input_size)
    if loaded is not None:
        print(f"  [no-retrain] {method_name} (fold {fold}) loaded from weights")
        y_pred, y_prob = loaded
        y_true = np.asarray(test_y).astype(int)
    else:
        y_true, y_pred, y_prob, save_dict = trainer()
        _save_weights(method_name, fold, save_dict)
    _record_splits(all_results, method_name, fold, y_true, y_pred, y_prob, country_arr)


def _training_cohort_sizes(msia_ds, india_ds, fold, num_of_folds=N_FOLDS_CV):
    """Pre-resampling training-row counts per cohort for one fold."""
    msia_df, msia_add_on, *_ = separate_df_and_df_add_on(msia_ds, LABEL, id_exists=True)
    india_df, india_add_on, *_ = separate_df_and_df_add_on(india_ds, LABEL, id_exists=False)
    validation_fold = (fold + 1) % num_of_folds
    msia_train, _ = process_raw_train_and_test_df(
        msia_df[msia_df["fold"] != validation_fold], msia_add_on, fold, id_exists=True)
    india_train, _ = process_raw_train_and_test_df(
        india_df[india_df["fold"] != validation_fold], india_add_on, fold, id_exists=False)
    return len(msia_train), len(india_train)


def _domain_labels(n_msia_train, n_india_train, n_train_total):
    """Domain vector for the pooled training rows, extended over the SMOTE rows."""
    domains = np.concatenate([np.zeros(n_msia_train), np.ones(n_india_train)])
    extra = n_train_total - (n_msia_train + n_india_train)
    if extra <= 0:
        return domains
    ratio = n_msia_train / (n_msia_train + n_india_train)
    extra_msia = int(extra * ratio)
    return np.concatenate([domains, np.zeros(extra_msia), np.ones(extra - extra_msia)])


def _arrays(prepared):
    """Model-ready float arrays for one prepared fold."""
    features = prepared.features
    return (
        prepared.train_df[features].values.astype(np.float32),
        prepared.train_df[LABEL].values.astype(np.int64),
        prepared.test_df[features].values.astype(np.float32),
        prepared.test_df[LABEL].values.astype(np.int64),
        np.asarray(prepared.country_arr, dtype=int),
    )


def run_experiment():
    """Train and score every domain-generalization comparator."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    common_msia, common_india = load_both_cohorts()
    harmonized_msia, harmonized_india = load_both_cohorts()
    size_msia, size_india = load_both_cohorts()

    # ARM A keeps nothing cross-domain (ten common features); ARM B uses the imputation-
    # quality gate.
    common_folds = build_harmonized_folds(
        common_msia, common_india, num_of_folds=N_FOLDS_CV,
        accuracy_threshold=ACCURACY_THRESHOLD, with_validation=True, selected_features=[],
    )
    harmonized_folds = build_harmonized_folds(
        harmonized_msia, harmonized_india, num_of_folds=N_FOLDS_CV,
        accuracy_threshold=ACCURACY_THRESHOLD, with_validation=True,
    )

    all_results = []
    for common_prepared, harmonized_prepared in zip(common_folds, harmonized_folds):
        fold = common_prepared.fold
        print(f"\nFold {fold}/{N_FOLDS_CV - 1}")

        print("\n  --- ARM A: Common Features (10 features, no imputation) ---")
        train_X, train_y, test_X, test_y, country_arr = _arrays(common_prepared)
        input_size = train_X.shape[1]
        n_msia_train, n_india_train = _training_cohort_sizes(size_msia, size_india, fold)
        domain_train = _domain_labels(n_msia_train, n_india_train, len(train_X))
        print(
            f"  Common train: {len(train_X)} x {input_size}; test: {len(test_X)} "
            f"(MY={int((country_arr == MALAYSIA).sum())}, "
            f"IN={int((country_arr == INDIA).sum())})"
        )

        print("  Training Naive FNN...")
        _evaluate_method(
            all_results, "Naive_FNN (10 common, ERM)", fold, test_X, test_y,
            input_size, country_arr,
            lambda: train_fnn(train_X, train_y, test_X, test_y, input_size),
        )
        print("  Training Naive LR...")
        _evaluate_method(
            all_results, "Naive_LR (10 common, ERM)", fold, test_X, test_y,
            input_size, country_arr,
            lambda: train_linear(train_X, train_y, test_X, test_y),
        )
        print("  Training CORAL+FNN...")
        _evaluate_method(
            all_results, "CORAL_FNN (10 common)", fold, test_X, test_y,
            input_size, country_arr,
            lambda: train_coral_fnn(train_X, train_y, domain_train, test_X, test_y,
                                    input_size),
        )
        print("  Training CORAL+LR...")
        _evaluate_method(
            all_results, "CORAL_LR (10 common)", fold, test_X, test_y,
            input_size, country_arr,
            lambda: train_coral_linear(train_X, train_y, domain_train, test_X, test_y),
        )
        print("  Training IRM+FNN...")
        _evaluate_method(
            all_results, "IRM_FNN (10 common)", fold, test_X, test_y,
            input_size, country_arr,
            lambda: train_irm_fnn(train_X, train_y, domain_train, test_X, test_y,
                                  input_size),
        )
        print("  Training DANN+FNN...")
        _evaluate_method(
            all_results, "DANN_FNN (10 common)", fold, test_X, test_y,
            input_size, country_arr,
            lambda: train_dann_fnn(train_X, train_y, domain_train, test_X, test_y,
                                   input_size),
        )

        print("\n  --- ARM B: Feature Harmonization (cross-imputation) ---")
        (harmonized_train_X, harmonized_train_y, harmonized_test_X,
         harmonized_test_y, harmonized_country) = _arrays(harmonized_prepared)
        harmonized_input_size = harmonized_train_X.shape[1]
        print(
            f"  Harmonized train: {len(harmonized_train_X)} x {harmonized_input_size}; "
            f"test: {len(harmonized_test_X)} "
            f"(MY={int((harmonized_country == MALAYSIA).sum())}, "
            f"IN={int((harmonized_country == INDIA).sum())})"
        )

        print("  Training Harmonized FNN (proposed)...")
        _evaluate_method(
            all_results, "Harmonized_FNN (proposed)", fold, harmonized_test_X,
            harmonized_test_y, harmonized_input_size, harmonized_country,
            lambda: train_fnn(harmonized_train_X, harmonized_train_y,
                              harmonized_test_X, harmonized_test_y,
                              harmonized_input_size),
        )
        print("  Training Harmonized LR (proposed)...")
        _evaluate_method(
            all_results, "Harmonized_LR (proposed)", fold, harmonized_test_X,
            harmonized_test_y, harmonized_input_size, harmonized_country,
            lambda: train_linear(harmonized_train_X, harmonized_train_y,
                                 harmonized_test_X, harmonized_test_y),
        )

        for row in [r for r in all_results
                    if r["fold"] == fold and r["eval_split"] == "total"]:
            print(
                f"    [total] {row['method']:<28} "
                f"BAcc={row['balanced_accuracy']:.4f} "
                f"AUC={row.get('roc_auc', float('nan')):.4f} F1={row['f1']:.4f}"
            )

    results_df = pd.DataFrame(all_results)
    per_fold_path = SAVE_DIR / "per_fold_results.csv"
    results_df.to_csv(per_fold_path, index=False)
    print(f"\nPer-fold results (with eval_split) saved to: {per_fold_path}")
    print(f"All model weights saved under: {WEIGHTS_DIR}/<method>/model_<fold>.(pt|pkl)")

    summary_df = summarize(results_df, group_cols=("method", "eval_split"))
    summary_path = SAVE_DIR / "summary_comparison_by_country.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary (method x country) saved to: {summary_path}")

    for split in EVAL_SPLITS:
        subset = summary_df[summary_df["eval_split"] == split]
        if len(subset) == 0:
            continue
        subset = subset.sort_values("roc_auc_mean", ascending=False)
        print(f"\nDG COMPARISON -- split: {split.upper()}  "
              f"(same FNN backbone; LR added for Naive & CORAL)")
        print(f"{'Method':<30} {'Bal. Acc.':<22} {'ROC AUC':<22} {'F1':<22} "
              f"{'Sens.':<22} {'Spec.':<22}")
        print("-" * 140)
        for _, row in subset.iterrows():
            print(
                f"{row['method']:<30} "
                f"{row.get('balanced_accuracy_str', 'N/A'):<22} "
                f"{row.get('roc_auc_str', 'N/A'):<22} "
                f"{row.get('f1_str', 'N/A'):<22} "
                f"{row.get('sensitivity_str', 'N/A'):<22} "
                f"{row.get('specificity_str', 'N/A'):<22}"
            )

    print("\nExperiment complete.")
    return results_df, summary_df


if __name__ == "__main__":
    set_seed(SEED)
    run_experiment()
