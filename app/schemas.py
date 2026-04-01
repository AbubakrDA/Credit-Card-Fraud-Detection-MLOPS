from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    """
    Pydantic schema representing the required features.
    
    Why this matters in MLOps:
    Without strict schema validation, bad inputs can reach the model inference 
    logic causing hard-to-debug crashes. Pydantic enforces typing before prediction.
    """
    Time: float = Field(..., description="Seconds since first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")

class PredictResponse(BaseModel):
    is_fraud: int
    probability: float
    threshold_used: float
    message: str
