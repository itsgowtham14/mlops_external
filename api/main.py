from pathlib import Path
import sys
import os
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.feature_engineering import add_features, FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "selected_model.joblib")))

app = FastAPI(
    title="Mango Remaining Shelf-Life API",
    description="Predicts remaining marketable shelf life in days for an arriving mango shipment.",
    version="1.0.0"
)
_model = None

class ShipmentInput(BaseModel):
    harvest_maturity_index: float = Field(ge=0, le=100)
    pulp_temperature_c: float = Field(ge=0, le=50)
    sweetness_brix: float = Field(ge=0, le=30)
    bruising_percentage: float = Field(ge=0, le=100)
    transport_duration_hours: float = Field(ge=0, le=200)
    cooling_delay_hours: float = Field(ge=0, le=72)
    packaging_type: Literal["ventilated_crate", "corrugated_carton", "plastic_crate"]

@app.on_event("startup")
def load_model():
    global _model
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict")
def predict(shipment: ShipmentInput):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    row = add_features(pd.DataFrame([shipment.model_dump()]))
    value = float(_model.predict(row[FEATURES])[0])
    return {"estimated_remaining_shelf_life_days": round(max(0.0, value), 1)}
