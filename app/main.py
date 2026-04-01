from fastapi import FastAPI, HTTPException
import pandas as pd
from app.schemas import PredictRequest, PredictResponse
from app.model_loader import load_champion_model
import os

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Inference API to detect fraudulent transactions using MLflow and Scikit-Learn pipelines.",
    version="1.0"
)

# Global model object (Loaded at startup to avoid reloading per request)
model = None

# A configurable decision threshold (Useful if we want to tilt towards higher recall dynamically)
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

@app.on_event("startup")
def startup_event():
    global model
    model = load_champion_model()
    if model is None:
        print("WARNING: Model failed to load. The /predict endpoint will return 500 errors.")

@app.get("/health")
def health_check():
    """
    Standard kubernetes health check ping.
    """
    return {
        "status": "Healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Accepts transaction data and returns fraud probability.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="ML model is not loaded into memory.")
        
    try:
        # Convert Pydantic request to Pandas DataFrame (required by sklearn Pipeline)
        data_df = pd.DataFrame([request.model_dump()])
        
        # Predict Probabilities
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(data_df)[0][1] # Probability of Class=1
            pred = 1 if prob >= DECISION_THRESHOLD else 0
        else:
            # Fallback if model doesn't support proba (e.g. strict SVM)
            pred = model.predict(data_df)[0]
            prob = float(pred)
            
        message = "Transaction identified as FRAUD" if pred == 1 else "Transaction identified as LEGITIMATE"
        
        return PredictResponse(
            is_fraud=pred,
            probability=prob,
            threshold_used=DECISION_THRESHOLD,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")
