"""Small FastAPI app for the Telco Customer Churn notebook."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", PROJECT_FOLDER / "models" / "churn_model.joblib")
)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_FOLDER / MODEL_PATH
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
LOW_RISK_LIMIT = 0.30
HIGH_RISK_LIMIT = 0.60


class CustomerFeatures(BaseModel):
    """The customer columns used by the fitted notebook pipeline."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=200, allow_inf_nan=False)
    TotalCharges: float = Field(ge=0, le=100000, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0, le=1)
    retention_probability: float = Field(ge=0, le=1)
    risk_category: Literal["Low", "Medium", "High"]
    model_version: str


def load_model(model_path: Path = MODEL_PATH):
    """Load the preprocessing and model pipeline saved by the notebook."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file was not found: {model_path}")
    return joblib.load(model_path)


def get_risk_category(churn_probability: float) -> str:
    """Low < 0.30, Medium 0.30 to < 0.60, High >= 0.60."""
    if churn_probability < LOW_RISK_LIMIT:
        return "Low"
    if churn_probability < HIGH_RISK_LIMIT:
        return "Medium"
    return "High"


def predict_customer(model, customer: CustomerFeatures) -> dict:
    customer_row = pd.DataFrame([customer.model_dump()])
    prediction = int(model.predict(customer_row)[0])
    positive_index = list(model.classes_).index(1)
    churn_probability = float(model.predict_proba(customer_row)[0, positive_index])
    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(churn_probability, 4),
        "retention_probability": round(1.0 - churn_probability, 4),
        "risk_category": get_risk_category(churn_probability),
        "model_version": MODEL_VERSION,
    }


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = load_model()
    yield
    application.state.model = None


app = FastAPI(
    title="Telco Customer Churn Predictor",
    description=(
        "Educational API for predicting churn from one customer record. "
        "Risk bands: Low < 0.30, Medium 0.30 to < 0.60, High >= 0.60."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "project": "Telco Customer Churn Predictor",
        "model_version": MODEL_VERSION,
        "docs": "/docs",
        "endpoints": ["GET /", "GET /health", "POST /predict"],
    }


@app.get("/health")
def health(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {
        "status": "ok" if model_loaded else "not_ready",
        "model_loaded": model_loaded,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return predict_customer(model, customer)
