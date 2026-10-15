"""Persistence of the per-fold data splits and DNN layer activations."""

from __future__ import annotations

import os


def save_train_test_set(train_test_dict, download_path):
    """Write every per-fold split in ``train_test_dict`` to CSV and Excel."""
    for split in ("csv/train", "excel/train", "csv/test", "excel/test"):
        os.makedirs(f"{download_path}/{split}", exist_ok=True)

    if "training_set" in train_test_dict.keys():
        for i, training_set in enumerate(train_test_dict["training_set"]):
            training_set.to_csv(
                f"{download_path}/csv/train/training_set_{i}.csv", index=False
            )
            training_set.to_excel(
                f"{download_path}/excel/train/training_set_{i}.xlsx", index=False
            )
    for i, testing_set in enumerate(train_test_dict["testing_set"]):
        testing_set.to_csv(f"{download_path}/csv/test/testing_set_{i}.csv", index=False)
        testing_set.to_excel(
            f"{download_path}/excel/test/testing_set_{i}.xlsx", index=False
        )
    for i, validation_set in enumerate(train_test_dict["validation_set"]):
        validation_set.to_csv(
            f"{download_path}/csv/test/validation_set_{i}.csv", index=False
        )
        validation_set.to_excel(
            f"{download_path}/excel/test/validation_set_{i}.xlsx", index=False
        )
    if "generated_set" in train_test_dict.keys():
        for i, generated_set in enumerate(train_test_dict["generated_set"]):
            generated_set.to_csv(
                f"{download_path}/csv/train/generated_set_{i}.csv", index=False
            )
            generated_set.to_excel(
                f"{download_path}/excel/train/generated_set_{i}.xlsx", index=False
            )
    if "raw_training_set" in train_test_dict.keys():
        for i, raw_training_set in enumerate(train_test_dict["raw_training_set"]):
            raw_training_set.to_csv(
                f"{download_path}/csv/train/raw_training_set_{i}.csv", index=False
            )
            raw_training_set.to_excel(
                f"{download_path}/excel/train/raw_training_set_{i}.xlsx", index=False
            )
    if "raw_testing_set" in train_test_dict.keys():
        for i, raw_testing_set in enumerate(train_test_dict["raw_testing_set"]):
            raw_testing_set.to_csv(
                f"{download_path}/csv/test/raw_testing_set_{i}.csv", index=False
            )
            raw_testing_set.to_excel(
                f"{download_path}/excel/test/raw_testing_set_{i}.xlsx", index=False
            )


def save_output_layer_dict(output_layer_dict_list, download_path):
    """Write the DNN hidden-layer activations of every fold."""
    os.makedirs(f"{download_path}/output_layers_weights/csv", exist_ok=True)
    os.makedirs(f"{download_path}/output_layers_weights/excel", exist_ok=True)

    for i, output_layer in enumerate(output_layer_dict_list):
        for key, df in output_layer.items():
            df.to_csv(
                f"{download_path}/output_layers_weights/csv/{key}_fold_{i}.csv",
                index=False,
            )
            df.to_excel(
                f"{download_path}/output_layers_weights/excel/{key}_fold_{i}.xlsx",
                index=False,
            )
