import pytest
import pandas as pd
from src.preprocess import build_preprocessor
from src.model_selection import select_best_model
from src.config import TARGET_COL

def test_preprocessing_pipeline_shape():
    """
    Validates that the scikit-learn pipeline correctly transforms data.
    
    Why this matters in MLOps:
    If a Transformer drops columns entirely or accidentally adds them, inference 
    will crash. This smoke test catches catastrophic data pipeline bugs.
    """
    # Create mock dataset
    df = pd.DataFrame({
        "Time": [3600, 7200],  # 1 hour and 2 hours
        "Amount": [10.0, 100.0],
        **{f"V{i}": [0.1, 0.2] for i in range(1, 29)},
        TARGET_COL: [0, 1]
    })
    
    X = df.drop(columns=[TARGET_COL])
    
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    # We should have Time + Amount + 28 PCA features = 30 features
    assert X_transformed.shape[1] == 30

def test_model_selection_logic():
    """
    Tests if the best-model selection properly triggers PR-AUC tie-breaker 
    and obeys Precision/Recall cutoffs.
    """
    
    # Model 1 fails recall cutoff (<0.75)
    # Model 2 passes both, good PR-AUC
    # Model 3 passes both, higher PR-AUC
    
    run_metrics = [
        {"run_id": "r1", "model_name": "A", "metrics": {"pr_auc": 0.99, "recall": 0.50, "precision": 0.90}},
        {"run_id": "r2", "model_name": "B", "metrics": {"pr_auc": 0.85, "recall": 0.80, "precision": 0.85}},
        {"run_id": "r3", "model_name": "C", "metrics": {"pr_auc": 0.90, "recall": 0.80, "precision": 0.85}},
    ]
    
    best_run = select_best_model(run_metrics)
    
    # We expect Model 3 (r3) to win. 
    # Even though r1 has the highest PR-AUC, it fails the recall business constraint.
    assert best_run == "r3"
