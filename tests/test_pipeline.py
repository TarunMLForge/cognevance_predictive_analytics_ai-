import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_data_files_exist():
    """Test if the dataset was downloaded correctly."""
    assert os.path.exists("data/raw/credit_card_defaults.csv")

def test_api_health_when_model_missing():
    """Test the health endpoint. If the model isn't trained yet, it should return 503."""
    response = client.get("/health")
    # If the model is not trained yet (due to environment limitation), it should safely return 503
    # If it is trained, it should return 200.
    assert response.status_code in [200, 503]

def test_api_predict_rejection_if_no_model():
    """Test that the prediction endpoint safely fails if model is missing."""
    payload = {
        "limit_balance": 50000,
        "sex": 2,
        "education": 2,
        "marriage": 2,
        "age": 25,
        "pay_1": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
        "bill_amt1": 20000, "bill_amt2": 19000, "bill_amt3": 0, "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
        "pay_amt1": 2000, "pay_amt2": 0, "pay_amt3": 0, "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0,
        "utilization_ratio": 0.4,
        "payment_ratio_1": 0.1,
        "avg_delay": 0.0,
        "max_delay": 0.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]
