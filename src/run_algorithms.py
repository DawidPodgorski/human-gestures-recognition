import time
from typing import Any
from sklearn import svm


def run_default(
    data_train: Any,
    data_val: Any,
    data_test: Any,
    labels_train: Any,
    labels_val: Any,
    labels_test: Any,
    SCALE_GAMMA_PARAM: float,
) -> dict[str, Any]:
    C_PARAM = 1.0
    model = svm.SVC(kernel="rbf", C=C_PARAM, gamma="scale")
    start_time = time.time()
    model.fit(data_train, labels_train)
    validation_accuracy = model.score(data_val, labels_val)
    end_time = time.time()
    elapsed_time = end_time - start_time
    test_accuracy = model.score(data_test, labels_test)

    return {
        "accuracy": float(validation_accuracy),
        "test_accuracy": float(test_accuracy),
        "elapsed_time": elapsed_time,
        "svm_params": [C_PARAM, SCALE_GAMMA_PARAM],
    }
