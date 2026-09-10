from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Credit Default Prediction API", description="API for predicting credit card default risk.")

MODEL_PATH = "models/best_model_pipeline.joblib"

# Load model globally (loads once at startup)
pipeline = None
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
else:
    print(f"Warning: Model not found at {MODEL_PATH}. Prediction endpoint will fail.")

# Define request body matching our engineered features
class ClientData(BaseModel):
    limit_balance: float
    sex: int
    education: int
    marriage: int
    age: int
    pay_1: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float
    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float
    utilization_ratio: float
    payment_ratio_1: float
    avg_delay: float
    max_delay: float

@app.get("/health")
def health_check():
    """Health check endpoint to verify API is running and model is loaded."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict_default(data: ClientData):
    """Predicts default probability for a single client."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not loaded.")
        
    try:
        # Convert Pydantic object to dict, then to DataFrame
        df_input = pd.DataFrame([data.model_dump()])
        
        # Make prediction
        prediction = pipeline.predict(df_input)[0]
        probability = pipeline.predict_proba(df_input)[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": "High" if probability > 0.5 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
