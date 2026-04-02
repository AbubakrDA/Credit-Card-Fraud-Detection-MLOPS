"""
Model evaluation utilities for imbalanced fraud classification.

In credit card fraud detection, accuracy is a misleading metric: a model that
always predicts 'Legitimate' achieves 99.83% accuracy while catching zero frauds.
This module computes metrics that reflect the true business cost of errors.
"""

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Compute a comprehensive set of fraud-relevant evaluation metrics.

    Metric Rationale:
        - Precision: Of all flagged frauds, how many were genuine?
                     Low precision → alert fatigue for the fraud operations team.
        - Recall:    Of all real frauds, how many did we detect?
                     Low recall → financial losses from undetected fraud.
        - PR-AUC:    Area under the Precision-Recall curve. Best holistic measure
                     for imbalanced binary classification. Used as the primary
                     model selection criterion.
        - ROC-AUC:   Supplementary metric; used for comparison across studies.
        - F1:        Harmonic mean of Precision and Recall; useful for reporting.

    Args:
        model:  A fitted Scikit-Learn compatible estimator with predict() method.
        X_test: Preprocessed test feature matrix.
        y_test: Ground-truth test labels (0=legitimate, 1=fraud).

    Returns:
        A tuple of:
            - metrics (Dict[str, float]): All computed metric scores.
            - cm (np.ndarray): 2x2 confusion matrix [[TN, FP], [FN, TP]].
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    metrics: Dict[str, float] = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_prob)),
        "pr_auc":    float(average_precision_score(y_test, y_prob)),
    }

    cm = confusion_matrix(y_test, y_pred)

    logger.info(
        "Evaluation | PR-AUC: %.4f | Recall: %.4f | Precision: %.4f",
        metrics["pr_auc"],
        metrics["recall"],
        metrics["precision"],
    )
    return metrics, cm
