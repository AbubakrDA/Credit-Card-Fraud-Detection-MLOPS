import pytest
from app.schemas import PredictRequest, PredictResponse
from pydantic import ValidationError

def test_valid_schema():
    """
    Validates that a correctly formatted payload parses into Pydantic successfully.
    
    Why this matters in MLOps:
    This protects the application logic from data drift or unexpected upstream 
    frontend changes.
    """
    payload = {
        "Time": 400.0,
        "Amount": 10.5,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    
    req = PredictRequest(**payload)
    assert req.Time == 400.0
    assert req.V1 == 0.0
    assert req.Amount == 10.5

def test_invalid_schema_missing_fields():
    """
    Checks if Pydantic properly blocks invalid requests.
    """
    payload = {
        "Time": 400.0
        # Missing all other fields
    }
    
    with pytest.raises(ValidationError):
        PredictRequest(**payload)
        
def test_invalid_schema_wrong_type():
    payload = {
        "Time": "Not a number",
        "Amount": 10.5,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    
    with pytest.raises(ValidationError):
        PredictRequest(**payload)
