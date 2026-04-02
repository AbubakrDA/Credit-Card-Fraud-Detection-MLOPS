"""
Feature engineering and preprocessing module.

Defines a custom Scikit-Learn transformer for cyclical time encoding and a
ColumnTransformer that scales Amount and passes PCA features through unchanged.
The entire preprocessor is Sklearn-compatible, meaning it serializes cleanly
into the MLflow artifact store and reproduces identically during API inference.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.config import AMOUNT_COL, PCA_COLS, TIME_COL

logger = logging.getLogger(__name__)


class TimeTransformer(BaseEstimator, TransformerMixin):
    """
    Converts the raw 'Time' column (seconds since first transaction) into
    the hour of day (0.0 – 23.99).

    Design Rationale:
        Raw elapsed seconds carry no predictive signal for unseen transactions
        because they are relative to a specific dataset's start time. Converting
        to 'hour of day' allows the model to learn temporal fraud patterns
        (e.g., high fraud rates at 2–4 AM) without data leakage.

        Encapsulating this as a Scikit-Learn Transformer ensures the same
        logic executes in training and production inference automatically.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "TimeTransformer":
        """No fitting required; transformation is stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply hour-of-day conversion.

        Args:
            X: DataFrame containing the TIME_COL column.

        Returns:
            Transformed DataFrame with TIME_COL replaced by hour of day.
        """
        X_out = X.copy()
        # 86400 seconds = 1 day; divide by 3600 to convert to hours
        X_out[TIME_COL] = (X_out[TIME_COL] % 86400) / 3600.0
        return X_out


def build_preprocessor() -> ColumnTransformer:
    """
    Construct the full Scikit-Learn feature preprocessing pipeline.

    Transformation Strategy:
        - Time:   TimeTransformer (hour-of-day) → RobustScaler
        - Amount: RobustScaler (handles large outliers like $25k transactions)
        - V1-V28: Passthrough (already unit-scaled by PCA in source dataset)

    Returns:
        An unfitted ColumnTransformer ready for `fit_transform`.
    """
    time_pipe = Pipeline(steps=[
        ("time_transform", TimeTransformer()),
        ("time_scale", RobustScaler()),
    ])

    amount_pipe = Pipeline(steps=[
        ("amount_scale", RobustScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("time",   time_pipe,     [TIME_COL]),
            ("amount", amount_pipe,   [AMOUNT_COL]),
            ("pca",    "passthrough", PCA_COLS),
        ],
        remainder="drop",  # Explicit: discard any unexpected columns
    )

    logger.info("Preprocessor built with features: time, amount, %d PCA cols", len(PCA_COLS))
    return preprocessor


def apply_preprocessing(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit the preprocessor on training data and transform both splits.

    Fitting on train-only prevents data leakage from the test set into
    the scaling statistics (mean, IQR).

    Args:
        preprocessor: An unfitted ColumnTransformer.
        X_train:      Training feature matrix.
        X_test:       Test feature matrix.

    Returns:
        Tuple of (X_train_preprocessed, X_test_preprocessed) as numpy arrays.
    """
    logger.info("Fitting preprocessor on training data...")
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    logger.info("Preprocessing complete. Output shape: %s", X_train_prep.shape)
    return X_train_prep, X_test_prep
