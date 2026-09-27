
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import json
import os


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Airbnb Price Prediction API",
    description="API for predicting Airbnb listing prices",
    version="1.0"
)


# --------------------------------------------------
# File Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_airbnb_price_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "feature_columns.json"
)


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# --------------------------------------------------
# Load Feature Columns
# --------------------------------------------------

with open(FEATURE_PATH, "r") as file:
    feature_columns = json.load(file)


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class PredictionRequest(BaseModel):
    data: dict


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Airbnb Price Prediction API is running",
        "endpoint": "/predict",
        "model": "Gradient Boosting Regressor"
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # Receive input JSON
        input_data = request.data

        # Convert JSON into DataFrame
        input_df = pd.DataFrame(
            [input_data]
        )

        # Add missing expected columns
        for column in feature_columns:

            if column not in input_df.columns:
                input_df[column] = None

        # Keep exactly the columns used during training
        input_df = input_df[
            feature_columns
        ]

        # Generate prediction
        prediction = model.predict(
            input_df
        )

        predicted_price = float(
            prediction[0]
        )

        return {
            "predicted_price": round(
                predicted_price,
                2
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
