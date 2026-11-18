from pathlib import Path
import joblib
import pandas as pd
from feature_engineering import add_features, FEATURES

ROOT = Path(__file__).resolve().parents[1]

def predict_one(shipment: dict, model_path=None) -> float:
    path = Path(model_path) if model_path else ROOT / "models" / "selected_model.joblib"
    model = joblib.load(path)
    row = add_features(pd.DataFrame([shipment]))
    prediction = float(model.predict(row[FEATURES])[0])
    return round(max(0.0, prediction), 1)
