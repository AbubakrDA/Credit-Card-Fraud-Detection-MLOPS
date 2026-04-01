import os
import mlflow.sklearn
from src.config import MLFLOW_TRACKING_URI

def load_champion_model():
    """
    Loads the best ML model from MLflow.
    
    Why this matters in MLOps:
    This prevents hardcoding a specific '.pkl' file path into our production code.
    If the data-science team trains a newer, better model, they just update the
    MLFLOW_MODEL_URI environment variable, and the container automatically uses 
    it without code changes.
    """
    # Ensure the loader uses the correct SQLite backend defined in config.py
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    model_uri = os.getenv("MLFLOW_MODEL_URI")
    
    if not model_uri:
        # Fallback for local testing
        print("WARNING: MLFLOW_MODEL_URI not found. This is acceptable for local testing.")
        return None
        
    print(f"Loading model from MLflow URI: {model_uri}")
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        return None
