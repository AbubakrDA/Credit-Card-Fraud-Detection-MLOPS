"""
Champion model selection with business constraint enforcement.

In production MLOps, the "best model" is not simply the one with the highest
metric. Real-world constraints — acceptable false positive rates, regulatory
recall minimums — must be evaluated first. This module encodes those rules.
"""

import logging
from typing import Any, Dict, List, Optional

from src.config import MIN_PRECISION_THRESHOLD, MIN_RECALL_THRESHOLD

logger = logging.getLogger(__name__)


def select_best_model(run_metrics_list: List[Dict[str, Any]]) -> str:
    """
    Select the champion model run ID using business-constrained optimization.

    Selection Algorithm:
        1. Filter candidates that satisfy BOTH hard constraints:
               Recall    >= MIN_RECALL_THRESHOLD    (default: 0.75)
               Precision >= MIN_PRECISION_THRESHOLD (default: 0.80)
        2. Among valid candidates, rank by PR-AUC (descending).
        3. If NO candidate satisfies the constraints, fall back to the highest
           PR-AUC across all candidates and log a warning.

    Args:
        run_metrics_list: List of dicts with structure:
            [
                {
                    "run_id": "abc123",
                    "model_name": "XGBoost",
                    "metrics": {"recall": 0.87, "precision": 0.82, "pr_auc": 0.91, ...}
                },
                ...
            ]

    Returns:
        The MLflow run_id (str) of the selected champion model.
    """
    valid_candidates = [
        entry for entry in run_metrics_list
        if (
            entry["metrics"]["recall"]    >= MIN_RECALL_THRESHOLD
            and entry["metrics"]["precision"] >= MIN_PRECISION_THRESHOLD
        )
    ]

    if valid_candidates:
        logger.info(
            "%d/%d model(s) passed business constraints (Recall≥%.2f, Precision≥%.2f).",
            len(valid_candidates),
            len(run_metrics_list),
            MIN_RECALL_THRESHOLD,
            MIN_PRECISION_THRESHOLD,
        )
        candidates = valid_candidates
    else:
        logger.warning(
            "No model met business constraints. Selecting by PR-AUC only."
        )
        candidates = run_metrics_list

    best = max(candidates, key=lambda x: x["metrics"]["pr_auc"])

    logger.info(
        "Champion selected: %s | Run ID: %s | PR-AUC: %.4f",
        best.get("model_name", "Unknown"),
        best["run_id"],
        best["metrics"]["pr_auc"],
    )
    return best["run_id"]
