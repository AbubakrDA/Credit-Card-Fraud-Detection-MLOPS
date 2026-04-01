import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "creditcard.csv"
# We use SQLite to enable the MLflow Model Registry locally.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR}/mlflow.db")

# Features
TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"
PCA_COLS = [f"V{i}" for i in range(1, 29)]

# Model Training Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Best Model Selection Rules
# These explicitly prioritize the constraints of credit card fraud detection
MIN_RECALL_THRESHOLD = 0.75
MIN_PRECISION_THRESHOLD = 0.80
