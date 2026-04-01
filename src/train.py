import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb

from src.config import RANDOM_STATE

def get_models():
    """
    Returns a dictionary of un-fitted models configured for imbalanced learning.
    
    Why this matters in MLOps:
    Centralized model definitions ensure we evaluate multiple algorithms on an
    equal playing field. `class_weight="balanced"` and `scale_pos_weight` are
    crucial: Since fraud makes up 0.17% of data, standard models will naturally
    ignore it to maximize accuracy. These weights heavily penalize missing frauds.
    """
    # For xgboost we use scale_pos_weight = count(negative) / count(positive)
    # Roughly 284315 / 492 ≈ 577
    xgboost_pos_weight = 577.0
    
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", 
            random_state=RANDOM_STATE, 
            max_iter=1000
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", 
            random_state=RANDOM_STATE, 
            n_estimators=100,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=xgboost_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="aucpr",
            n_estimators=100,
            n_jobs=-1
        )
    }
    
    return models

def train_model(model_name: str, model, X_train, y_train):
    """
    Fits a given model.
    """
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    return model
