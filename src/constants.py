from typing import Optional

# test data preprocessing
TEST_DATA_PATH: str = "datasets/gestureData/testData/Data_For_Test_Random.mat"
CACHE_PATH: str = "datasets/gestureData/testData/Data_For_Test_Random.pkl"
NUMERICAL_VERSION_OF_LABELS: dict[str, int] = {
    "Wave": 0,
    "Pinch": 1,
    "Swipe": 2,
    "Click": 3,
}
TEXT_VERSION_OF_LABELS: dict[int, str] = {
    0: "Wave",
    1: "Pinch",
    2: "Swipe",
    3: "Click",
}

TARGET_TEST_DATASET_PATH_SAVE: str = (
    "datasets/gestureData/testData/target_testing_dataset.pkl"
)
TARGET_TEST_DATASET_PATH_LOAD: str = (
    "datasets/gestureData/testData/target_testing_dataset.pkl"
)

# training data preprocessing
TRAINING_DATA_PATH: str = (
    "datasets/gestureData/trainingData/Data_Per_PersonData_Training_Person_"
)
TRAINING_CACHE_PATH: str = (
    "datasets/gestureData/trainingData/Data_Per_PersonData_Training_Person_"
)
PERSONS: list[str] = ["A", "B", "C", "D", "E", "F"]
TARGET_TRAINING_DATASET_PATH_SAVE: str = (
    "datasets/gestureData/trainingData/target_training_dataset.pkl"
)
TARGET_TRAINING_DATASET_PATH_LOAD: str = (
    "datasets/gestureData/trainingData/target_training_dataset.pkl"
)


# common data preprocessing
TIME_BINS: int = 15
FREQUENCY_BINS: int = 15
# NUMBER_OF_FEATURES is equal to ((TIME_BINS // 2 + 1) * FREQUENCY_BINS)  this is formula for basic extract features
IS_PCA: bool = False
IS_DEBUG_MODE: bool = False
NUMBER_OF_FEATURES: int = 643  # for PCA
TRESHOLD_DB: Optional[int] = None

# svm models
SVM_MODEL_PATH: str = "svmModels/"
