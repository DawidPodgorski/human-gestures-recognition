from typing import Any

import numpy as np
from dataPreprocessing.get_test_dataset_for_svm import get_test_dataset_for_svm
from dataPreprocessing.get_training_dataset_for_svm import get_training_dataset_for_svm
from run_algorithms import run_default
from sklearn.model_selection import train_test_split
from tools import get_gamma_param, save_data_pickle
import constants
from datetime import datetime
from sklearn import svm


class Experiment:
    def __init__(self) -> None:
        self.data: np.ndarray
        self.labels: np.ndarray

        self.data, self.labels, self.test_data, self.test_labels = self._load_dataset()
        (
            self.data_train,
            self.data_validation,
            self.labels_train,
            self.labels_validation,
        ) = train_test_split(self.data, self.labels, test_size=0.4, random_state=42)

        self.default_results: dict[str, Any] = {}

    @staticmethod
    def _load_dataset() -> tuple[np.ndarray, np.ndarray]:
        train_dataset = get_training_dataset_for_svm()
        train_data = train_dataset["data"]
        train_labels = train_dataset["labels"]
        test_dataset = get_test_dataset_for_svm()
        test_data = test_dataset["data"]
        test_labels = test_dataset["labels"]

        print(f"data.shape={train_data.shape}, labels.shape={train_labels.shape}\n")
        return train_data, train_labels, test_data, test_labels

    def _print_information(self):
        print("Default SVM will be running")

    def _collect_algorithms_results(
        self,
    ):

        scale_gamma_param = get_gamma_param(
            n_features=self.data.shape[1], x_var=self.data.var()
        )
        self.default_results = run_default(
            self.data_train,
            self.data_validation,
            self.test_data,
            self.labels_train,
            self.labels_validation,
            self.test_labels,
            scale_gamma_param,
        )

    def _save_svm_model(
        self, svm_params: list[float], svm_algorithm_name: str = "default"
    ) -> None:
        svm_model = svm.SVC(kernel="rbf", C=svm_params[0], gamma=svm_params[1])
        svm_model.fit(self.data, self.labels)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_data_pickle(
            svm_model,
            constants.SVM_MODEL_PATH
            + svm_algorithm_name
            + "_svm_model"
            + current_datetime
            + ".pkl",
        )

    def _print_results_for_default(self):
        print("Default SVM Results:")
        print(f"Validation Accuracy: {self.default_results['accuracy']:.4f}")
        print(f"Test Accuracy: {self.default_results['test_accuracy']:.4f}")
        print(f"Elapsed Time: {self.default_results['elapsed_time']:.4f} seconds")
        print(f"SVM Parameters (C, gamma): {self.default_results['svm_params']}\n")

    def run_experiment(
        self,
    ):
        self._print_information()
        self._collect_algorithms_results()
        user_input = (
            input(f"Do you want to save the svm model? (y/n): ").strip().lower()
        )
        if user_input == "y":
            self._save_svm_model(self.default_results["svm_params"])
        self._print_results_for_default()
