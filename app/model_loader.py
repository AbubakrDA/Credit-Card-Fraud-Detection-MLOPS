"""
MLflow model loading module for the FastAPI inference service.

Decouples the serving layer from the training layer by loading models
exclusively via the MLflow Model Registry URI, not from local file paths.
This means the fraud detection API can be upgraded to a new champion model
with zero code changes — only an environment variable update is needed.
"""

import logging
import os
from typing import Any, Optional

import mlflow.sklearn

from src.config import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)


def load_champion_model() -> Optional[Any]:
    """
    Load the champion fraud detection model from the MLflow Model Registry.

    The model URI is read from the MLFLOW_MODEL_URI environment variable,
    which supports both local SQLite registries and remote HTTP servers:

        Local (dev):       models:/FraudDetector/latest
        Docker (prod):     models:/FraudDetector/latest
                           (with MLFLOW_TRACKING_URI=http://mlflow:5000)

    Returns:
        A fitted Scikit-Learn pipeline if loading succeeds, or None if the
        model URI is not configured (acceptable in unit test environments).

    Raises:
        Does not raise. All exceptions are caught and logged to avoid
        crashing the API container on startup.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = os.getenv("MLFLOW_MODEL_URI")

    if not model_uri:
        logger.warning(
            "MLFLOW_MODEL_URI is not set. "
            "The /predict endpoint will be unavailable until a model is configured."
        )
        return None

    logger.info("Loading champion model from registry URI: %s", model_uri)
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully. Type: %s", type(model).__name__)
        return model
    except Exception as exc:
        logger.error(
            "Failed to load model from MLflow. URI: %s | Error: %s",
            model_uri,
            exc,
        )
        return None
