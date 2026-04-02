"""
Data ingestion and splitting module for the Credit Card Fraud Detection pipeline.

Centralizes all I/O operations so that training scripts, notebooks, and DAGs
always consume data from the same authoritative source.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TEST_SIZE, RANDOM_STATE, TARGET_COL

logger = logging.getLogger(__name__)


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load the raw credit card fraud CSV dataset.

    Args:
        filepath: Absolute or relative path to the CSV file.

    Returns:
        Raw DataFrame with all original columns intact.

    Raises:
        FileNotFoundError: If the dataset file does not exist at the given path.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    logger.info(
        "Dataset loaded: %d rows | Fraud rate: %.4f%%",
        len(df),
        df[TARGET_COL].mean() * 100,
    )
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split on the fraud dataset.

    Stratification is critical here because the fraud class represents
    only ~0.17% of all transactions. Without it, test sets may contain
    no fraud samples at all, making evaluation impossible.

    Args:
        df: Full DataFrame including the target column.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # Essential for preserving fraud ratio in both splits
    )

    logger.info(
        "Data split complete | Train: %d | Test: %d | Test fraud rate: %.4f%%",
        len(X_train),
        len(X_test),
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test
