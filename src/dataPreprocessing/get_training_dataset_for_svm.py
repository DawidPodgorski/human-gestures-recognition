import os

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


def _load_all_data(mat_file_path: str, cache_file_path: str) -> dict[str, np.ndarray]:
    all_data: dict[str, np.ndarray] = {}
    for person in constants.PERSONS:
        absolute_path_to_data = mat_file_path + person + ".mat"
        absolute_path_to_cache = cache_file_path + person + ".pkl"
        all_data[person] = _load_data(absolute_path_to_data, absolute_path_to_cache)
    return all_data


def _load_data(mat_file_path: str, cache_file_path: str) -> np.ndarray:
    if os.path.exists(cache_file_path):
        print("Loading base data from cache...")
        data = load_data_pickle(cache_file_path)
    else:
        print("Loading base data from .mat file...")
        data = sio.loadmat(mat_file_path)
        save_data_pickle(data, cache_file_path)

    target_test_data: np.ndarray = data["Data_Training"]["Doppler_Signals"][0][0][0]
    return target_test_data


def _extract_target_data(test_data_sample: np.ndarray) -> np.ndarray:
    target_test_data_sample: np.ndarray = test_data_sample[0]
    return target_test_data_sample


def _create_data_set(
    all_test_data: dict[str, np.ndarray],
    number_of_features: int,
    is_pca: bool = False,
    is_debug_mode: bool = False,
) -> dict[str, np.ndarray]:

    dataset: dict[str, np.ndarray] = {}
    dataset["data"] = []
    dataset["labels"] = []
    total_samples = sum(
        len(gesture_samples)
        for person_data in all_test_data.values()
        for gesture_samples in person_data
    )
    with tqdm(total=total_samples, desc="Processing Data", unit="sample") as pbar:
        for person_data in all_test_data.values():
            for gesture_label, gesture_samples in enumerate(person_data):
                for sample_data in gesture_samples:
                    target_data_sample = _extract_target_data(sample_data)
                    optimized_target_data_sample = optimize_data(target_data_sample)
                    features = extract_features_from_spectrogram_advanced(
                        optimized_target_data_sample,
                    )
                    if is_debug_mode:
                        display_inforamtion_about_data(
                            gesture_label, features, optimized_target_data_sample
                        )
                    dataset["data"].append(features)
                    dataset["labels"].append(gesture_label)
                    pbar.update(1)
    dataset["data"] = np.array(dataset["data"], dtype=np.float64)
    dataset["labels"] = np.array(dataset["labels"], dtype=np.int64)

    if is_pca:
        dataset["data"] = apply_pca(
            dataset["data"], dataset["labels"], number_of_features
        )
    dataset["data"] = normalize_features(dataset["data"])

    return dataset


def get_training_dataset_for_svm() -> dict[str, np.ndarray]:
    if_load_data: bool = (
        input("Do you want to load target data from cache? (y/n): ") == "y"
    )
    if os.path.exists(constants.TARGET_TRAINING_DATASET_PATH_LOAD) and if_load_data:
        print("Loading target data from cache...")
        target_dataset: dict[str, np.ndarray] = load_data_pickle(
            constants.TARGET_TRAINING_DATASET_PATH_LOAD
        )
    else:
        print("Creating target data...")
        all_test_data: dict[str, np.ndarray] = _load_all_data(
            constants.TRAINING_DATA_PATH,
            constants.TRAINING_CACHE_PATH,
        )
        target_dataset: dict[str, np.ndarray] = _create_data_set(
            all_test_data,
            constants.NUMBER_OF_FEATURES,
            is_pca=constants.IS_PCA,
            is_debug_mode=constants.IS_DEBUG_MODE,
        )
        print("Saving target data...")
        save_data_pickle(target_dataset, constants.TARGET_TRAINING_DATASET_PATH_SAVE)
    return target_dataset


def main():
    target_data = get_training_dataset_for_svm()
    print(target_data["data"].shape)


if __name__ == "__main__":
    main()
