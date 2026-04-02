"""
Model definitions and training utilities.

Centralizes all classifier configurations in one place, ensuring every model
is trained under identical conditions and with appropriate imbalance handling.
"""

import logging
from typing import Any, Dict

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)

# Ratio of negative to positive class (≈ 284315 / 492 ≈ 577).
# Used by XGBoost to apply asymmetric loss weighting.
_XGBOOST_SCALE_POS_WEIGHT: float = 577.0


def get_models() -> Dict[str, Any]:
    """
    Return a dictionary of pre-configured, unfitted classifiers.

    All models are configured to handle extreme class imbalance (0.17% fraud):
      - Logistic Regression / Random Forest: `class_weight='balanced'` scales
        sample weights inversely proportional to class frequencies.
      - XGBoost: `scale_pos_weight` upweights the minority (fraud) class by
        the negative/positive sample ratio (~577).

    Returns:
        Dict mapping model name (str) to an unfitted classifier instance.
    """
    models: Dict[str, Any] = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_estimators=100,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=_XGBOOST_SCALE_POS_WEIGHT,
            random_state=RANDOM_STATE,
            eval_metric="aucpr",
            n_estimators=100,
            n_jobs=-1,
            verbosity=0,
        ),
    }
    logger.info("Initialized %d model(s): %s", len(models), list(models.keys()))
    return models


def train_model(model_name: str, model: Any, X_train, y_train) -> Any:
    """
    Fit a single classifier on the preprocessed training data.

    Args:
        model_name: Human-readable name used for logging (e.g. 'XGBoost').
        model:      An unfitted Scikit-Learn compatible estimator.
        X_train:    Preprocessed training feature matrix.
        y_train:    Training target labels (0=legit, 1=fraud).

    Returns:
        The fitted estimator.
    """
    logger.info("Training %s...", model_name)
    model.fit(X_train, y_train)
    logger.info("%s training complete.", model_name)
    return model
