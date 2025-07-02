import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from src.api.pydantic_models import PredictionInput, PredictionOutput

app = FastAPI(title="Credit Risk Prediction API")

# Load the best model from MLflow Model Registry
model_name = "CreditRisk_LogisticRegression"  # Update with your registered model name
model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    # Convert input data to DataFrame
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    
    # Predict risk probability
    prob = model.predict_proba(df)[:, 1][0]
    
    return PredictionOutput(risk_probability=prob)