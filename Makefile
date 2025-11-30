# Define the Python interpreter
PYTHON := python3

# Source directory for the project
SRC_DIR := src
DATA_DIR := datasets
# Define PYTHONPATH for the project
export PYTHONPATH := $(SRC_DIR):$(DATA_DIR)

.PHONY: run
run:
	$(PYTHON) $(SRC_DIR)/main.py

.PHONY: testData
testData:
	$(PYTHON) $(SRC_DIR)/dataPreprocessing/get_test_dataset_for_svm.py

.PHONY: trainingData
trainingData:
	$(PYTHON) $(SRC_DIR)/dataPreprocessing/get_training_dataset_for_svm.py

# Target: Clean up temporary files (e.g., .pyc, __pycache__)
.PHONY: clean
clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +