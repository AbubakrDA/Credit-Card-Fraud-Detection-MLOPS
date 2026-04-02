"""
Project-wide configuration for the Credit Card Fraud Detection MLOps platform.

All paths, ML parameters, and tracking URIs are centralized here to ensure
consistency across training, evaluation, and serving components.
"""

import os
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = BASE_DIR / "creditcard.csv"

# ---------------------------------------------------------------------------
# DATASET COLUMNS
# ---------------------------------------------------------------------------
TARGET_COL: str = "Class"
TIME_COL: str = "Time"
AMOUNT_COL: str = "Amount"
PCA_COLS: list = [f"V{i}" for i in range(1, 29)]

# ---------------------------------------------------------------------------
# TRAINING CONFIGURATION
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# MLFLOW / MODEL REGISTRY
# ---------------------------------------------------------------------------
EXPERIMENT_NAME: str = "Credit_Card_Fraud_Detection"

# Supports both local SQLite (Windows dev) and remote HTTP (Docker/Production)
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{BASE_DIR}/mlflow.db"
)

# ---------------------------------------------------------------------------
# BUSINESS LOGIC THRESHOLDS
# ---------------------------------------------------------------------------
# These enforce fraud-domain constraints before selecting the champion model.
# Recall >= 0.75: We cannot miss too many real fraud cases (false negatives).
# Precision >= 0.80: We cannot flag too many legit transactions (false positives).
MIN_RECALL_THRESHOLD: float = 0.75
MIN_PRECISION_THRESHOLD: float = 0.80

logger.info("Project configuration loaded. Base directory: %s", BASE_DIR)
