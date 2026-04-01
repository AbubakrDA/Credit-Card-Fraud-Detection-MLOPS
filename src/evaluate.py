from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix
)
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Computes rigorous evaluation metrics for imbalanced classification.
    
    Why this matters in MLOps:
    In Credit Card Fraud, accuracy is irrelevant. If 99.8% of cases are legit, 
    a dumb model predicting "legit" always is 99.8% accurate but entirely useless.
    - Precision: "When we flag a fraud, are we right?" (Low precision means alert fatigue).
    - Recall: "Out of all real frauds, how many did we catch?" (This is paramount).
    - PR-AUC: Measures area under precision-recall curve. Highest value means the 
              model operates effectively without dropping performance at different thresholds.
    """
    # 1. Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # 2. Compute Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc)
    }
    
    return metrics, cm
