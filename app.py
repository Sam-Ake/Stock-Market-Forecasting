import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Stock Market Time-Series Forecasting API",
    description=(
        "API endpoints for stock price forecasting using lag-based Random"
        " Forest regression."
    ),
    version="1.0.0",
)


class ForecastInput(BaseModel):
  lag_1: float
  lag_2: float
  lag_3: float
  lag_4: float
  lag_5: float
  rolling_mean_7: float
  rolling_mean_30: float


model = None


@app.on_event("startup")
def load_model():
  global model
  try:
    model = joblib.load("stock_rf_model.pkl")
    print("Loaded trained Random Forest model successfully.")
  except Exception as e:
    print(f"⚠️ Error loading model: {e}")


@app.get("/")
def home():
  return {
      "message": "Welcome to the Stock Forecast API! Use /docs to test endpoints."
  }


@app.post("/forecast")
def predict_stock(data: ForecastInput):
  if model is None:
    raise HTTPException(
        status_code=500, detail="Model is not loaded on the server."
    )

  try:
    features = np.array([[
        data.lag_1,
        data.lag_2,
        data.lag_3,
        data.lag_4,
        data.lag_5,
        data.rolling_mean_7,
        data.rolling_mean_30,
    ]])

    prediction = model.predict(features)[0]

    return {
        "status": "success",
        "predicted_close_price": round(float(prediction), 4),
    }
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))