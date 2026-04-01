from src.config import MIN_PRECISION_THRESHOLD, MIN_RECALL_THRESHOLD

def select_best_model(run_metrics_list: list) -> str:
    """
    Selects the best model based on explicit fraud detection business logic.
    
    Input:
        run_metrics_list: List of dictionaries e.g.
        [{"run_id": "1", "model": "XGB", "metrics": {"pr_auc": 0.8, "recall": 0.9, "precision": 0.82}}, ...]
        
    Why this matters in MLOps:
    A random metric optimization doesn't tie back to business reality.
    We need high PR-AUC, but we explicitly require minimum Precision & Recall 
    standards so the fraud team isn't overwhelmed (precision barrier) but doesn't 
    miss too much (recall barrier). If multiple models pass the barrier, the one 
    with the highest PR-AUC wins. If none do, we pick the highest PR-AUC natively.
    """
    
    valid_candidates = []
    
    for entry in run_metrics_list:
        metrics = entry["metrics"]
        
        # Check against constraints
        if metrics["recall"] >= MIN_RECALL_THRESHOLD and metrics["precision"] >= MIN_PRECISION_THRESHOLD:
            valid_candidates.append(entry)
            
    # Default to all if none passed thresholds
    candidates_to_score = valid_candidates if valid_candidates else run_metrics_list
    
    # Select by highest PR_AUC
    best_candidate = sorted(candidates_to_score, key=lambda x: x["metrics"]["pr_auc"], reverse=True)[0]
    
    return best_candidate["run_id"]
