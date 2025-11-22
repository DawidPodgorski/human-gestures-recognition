import os
import pickle
from typing import Optional

import constants
import numpy as np
import scipy.io as sio
from dataPreprocessing.extract_features_from_spectrogram import (
    apply_pca,
    extract_features_from_spectrogram_advanced,
    normalize_features,
)
from tools import (
    display_inforamtion_about_data,
    load_data_pickle,
    optimize_data,
    save_data_pickle,
)
from tqdm import tqdm


def _load_data(mat_file_path: str, cache_file_path: str) -> np.ndarray:
    if os.path.exists(cache_file_path):
        print("Loading base data from cache...")
        with open(cache_file_path, "rb") as f:
            data = pickle.load(f)
    else:
        print("Loading base data from .mat file...")
        data = sio.loadmat(mat_file_path)

        with open(cache_file_path, "wb") as f:
            pickle.dump(data, f)

    target_test_data: np.ndarray = data["Data_rand"]
    return target_test_data


def _get_label(test_data_sample: np.ndarray) -> str:
    label: str = str(test_data_sample[1][0]).split()[0]
    return label


def _extract_target_data_and_label(
    test_data_sample: np.ndarray,
) -> tuple[np.ndarray, str]:
    target_test_data_sample: np.ndarray = test_data_sample[0][0][0]
    label: str = _get_label(test_data_sample)
    return target_test_data_sample, label


def _create_data_set(
    test_data: np.ndarray,
    number_of_features: int,
    is_pca: bool = False,
    is_debug_mode: bool = False,
) -> dict[str, np.ndarray]:

    dataset: dict[str, np.ndarray] = {}
    dataset["data"] = []
    dataset["labels"] = []

    for data_sample in tqdm(test_data, desc="Processing Data", unit="sample"):
        target_data_sample, label = _extract_target_data_and_label(data_sample)
        optimized_data = optimize_data(target_data_sample)
        numerical_label = constants.NUMERICAL_VERSION_OF_LABELS[label]
        features = extract_features_from_spectrogram_advanced(optimized_data)
        dataset["labels"].append(numerical_label)
        dataset["data"].append(features)
        if is_debug_mode:
            display_inforamtion_about_data(numerical_label, features, optimized_data)
    dataset["data"] = np.array(dataset["data"], dtype=np.float64)
    dataset["labels"] = np.array(dataset["labels"], dtype=np.int64)

    if is_pca:
        dataset["data"] = apply_pca(
            dataset["data"], dataset["labels"], number_of_features
        )
    dataset["data"] = normalize_features(dataset["data"])

    return dataset


def get_test_dataset_for_svm(
    get_new_test_data: Optional[bool] = None,
) -> dict[str, np.ndarray]:
    if get_new_test_data is None:
        if_load__target_data_from_cache: bool = (
            input("Do you want to load target data from cache? (y/n): ") == "y"
        )
    else:
        if_load__target_data_from_cache: bool = False

    if (
        os.path.exists(constants.TARGET_TEST_DATASET_PATH_LOAD)
        and if_load__target_data_from_cache
    ):
        print("Loading target data from cache...")
        target_test_data: dict[str, np.ndarray] = load_data_pickle(
            constants.TARGET_TEST_DATASET_PATH_LOAD
        )
    else:
        print("Creating target data...")
        test_data: np.ndarray = _load_data(
            constants.TEST_DATA_PATH, constants.CACHE_PATH
        )
        target_test_data = _create_data_set(
            test_data,
            constants.NUMBER_OF_FEATURES,
            is_pca=constants.IS_PCA,
            is_debug_mode=constants.IS_DEBUG_MODE,
        )
        print("Saving target data...")
        save_data_pickle(target_test_data, constants.TARGET_TEST_DATASET_PATH_SAVE)
    return target_test_data


def main():
    target_data = get_test_dataset_for_svm()
    print(target_data["data"].shape)


if __name__ == "__main__":
    main()
