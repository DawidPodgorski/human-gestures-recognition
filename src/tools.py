import os
import pickle
from typing import Any

import constants
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def get_gamma_param(n_features: int, x_var: float) -> float:
    return 1 / (n_features * x_var)


def optimize_data(target_data_sample: np.ndarray) -> np.ndarray:
    spectrogram_magnitude_db: np.ndarray = 20 * np.log10(
        abs(target_data_sample) / np.amax(abs(target_data_sample))
    )
    if constants.TRESHOLD_DB is not None:
        spectrogram_magnitude_db = np.where(
            spectrogram_magnitude_db >= constants.TRESHOLD_DB,
            spectrogram_magnitude_db,
            0,
        )
    return spectrogram_magnitude_db


def display_spectrogram(
    spectrogram_target_data: np.ndarray, spectrogram_title: str = ""
) -> None:
    plt.imshow(
        spectrogram_target_data,
        vmin=-50,
        vmax=0,
        cmap="jet",
        aspect="auto",
    )
    plt.colorbar()
    plt.ylabel("Doppler", fontsize=17)
    plt.xlabel("Time", fontsize=17)
    plt.title(spectrogram_title)
    plt.show()


def load_dataset_np(file_path: str) -> dict[str, np.ndarray]:
    data = np.load(file_path)
    dataset = {"data": data["data"], "labels": data["labels"]}
    return dataset


def save_dataset_np(dataset: dict[str, np.ndarray], file_path: str):
    np.savez(file_path, data=dataset["data"], labels=dataset["labels"])


def get_new_file_name(file_path: str) -> str:
    file_name, file_extension = os.path.splitext(file_path)

    counter = 1
    new_file_path = f"{file_name}_{counter}{file_extension}"

    while os.path.exists(new_file_path):
        counter += 1
        new_file_path = f"{file_name}_{counter}{file_extension}"

    return new_file_path


def save_data_pickle(data: Any, file_path: str):
    if os.path.exists(file_path):
        user_input = (
            input(
                f"The file '{file_path}' already exists. Do you want to overwrite it? (y/n): "
            )
            .strip()
            .lower()
        )
        if user_input == "n":
            file_path = get_new_file_name(file_path)
            print(f"Saving with a new file name: '{file_path}'")

    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Data has been saved to '{file_path}'.")


def load_data_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def display_inforamtion_about_data(
    label: int, data_sample: np.ndarray, optimized_data_sample: np.ndarray
) -> None:
    print(f"label: {constants.TEXT_VERSION_OF_LABELS[label]}")
    print(f"feature shape: {data_sample.shape}")
    print(f"optimized data shape: {optimized_data_sample.shape}")
    display_spectrogram(optimized_data_sample, constants.TEXT_VERSION_OF_LABELS[label])


def show_confusion_matrix(test_data: list[int], predictions: list[int], model_name=""):
    cm = confusion_matrix(test_data, predictions, normalize="true")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(constants.TEXT_VERSION_OF_LABELS.values()),
    )
    disp.plot(cmap=plt.cm.Blues, values_format=".1%")

    plt.title(f"Confusion matrix for {model_name}")
    plt.show()
