import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """
    Tests if the API boots successfully.
    
    Why this matters in MLOps:
    Health checks are the core mechanism that Kubernetes/Docker uses to 
    determine if a pod should be killed or routed traffic.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "Healthy"

def test_predict_endpoint_no_model():
    """
    Since we don't naturally load a Model into the test environment via MLflow, 
    the API should correctly detect that and throw a 500 error instead of failing silently.
    """
    payload = {
        "Time": 1.0, "Amount": 10.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    
    response = client.post("/predict", json=payload)
    # 500 expects server state error (model not loaded)
    assert response.status_code == 500
    assert "ML model is not loaded" in response.json()["detail"]
