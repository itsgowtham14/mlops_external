# Mango Remaining Shelf-Life Prediction (MLOps)

End-to-end educational MLOps project for estimating the remaining marketable shelf life (days) of mango shipments arriving at a packing facility near Nuzvid, Andhra Pradesh.

> **Important:** The datasets are synthetic and the model is a demonstration. Do not use its predictions for commercial export decisions until it has been trained, calibrated and independently validated on real cultivar-, farm-, season- and cold-chain-specific observations.

## Pipeline
1. Generate two versioned synthetic datasets (V1 baseline; V2 with longer transport/cooling delays).
2. Engineer reusable heat exposure and handling damage features.
3. Train Random Forest and Gradient Boosting regressors; compare MAE, RMSE and R².
4. Track runs and register models in MLflow.
5. Save the selected pipeline for FastAPI serving.
6. Run input-validation and prediction tests locally and in GitHub Actions.

## Setup (Python 3.11 recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run in order
```bash
python src/generate_data.py --rows 300
python src/preprocessing.py
python src/train.py
pytest -q
uvicorn api.main:app --reload
```
Open http://127.0.0.1:8000/docs for interactive API documentation and http://127.0.0.1:8000/health for health status.

## Example request
POST `/predict`:
```json
{
  "harvest_maturity_index": 85,
  "pulp_temperature_c": 24,
  "sweetness_brix": 14.5,
  "bruising_percentage": 4,
  "transport_duration_hours": 8,
  "cooling_delay_hours": 2,
  "packaging_type": "corrugated_carton"
}
```
Response:
```json
{"estimated_remaining_shelf_life_days": 12.3}
```
The number above is an illustrative response shape, not a guaranteed model output. Your actual result depends on the generated data and training run.

## MLflow
Training creates a local SQLite tracking database (`mlflow.db`) and logs both candidates. It registers each trained candidate under `MangoShelfLifeModel`. Start the UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```
Visit http://127.0.0.1:5000. Compare MAE/RMSE/R² and identify the selected model from `models/selected_model.txt` and `models/evaluation.json`. The API loads the selected serialized pipeline from `models/selected_model.joblib`.

## Dataset schema
- `harvest_maturity_index`: synthetic 0–100 maturity scale
- `pulp_temperature_c`: pulp temperature at arrival, °C
- `sweetness_brix`: soluble solids estimate, °Brix
- `bruising_percentage`: estimated visibly bruised surface percentage
- `transport_duration_hours`: time from harvest to facility, hours
- `cooling_delay_hours`: delay before cooling, hours
- `packaging_type`: ventilated_crate / corrugated_carton / plastic_crate
- `remaining_shelf_life_days`: synthetic target
- `dataset_version`: v1 / v2

V1 and V2 each contain 300 rows by default (600 total). V2 uses higher average transport duration and cooling delay distributions. The generated CSVs are reproducible using fixed random seeds.

## Engineered features
- `heat_exposure_index = max(pulp_temperature_c - 18, 0) × transport_duration_hours / 24`
- `handling_damage_index = bruising_percentage × 0.10 + cooling_delay_hours × 0.32`

These are simple demonstration proxies. Their coefficients and the 18°C reference threshold should be validated with domain experts and real postharvest data.

## CI/CD
GitHub Actions runs `pytest` on pushes and pull requests to `main` or `master`. Tests check Pydantic input boundaries, engineered-feature creation and that a model can execute a prediction. The workflow validates code; it does not automatically retrain or deploy the model. For production, add a controlled training workflow, model approval, artifact storage, and deployment job.

## Versioning and reproducibility
The raw files are named `mango_v1.csv` and `mango_v2.csv`; processed features have matching version names. The generator uses fixed seeds. Commit the scripts and optionally the generated CSVs if repository size permits. For formal data lineage, add DVC or a managed feature store; the reusable CSV feature files here are a lightweight feature-store demonstration.

## Limitations and production recommendations
Use real observations with a clearly defined marketability endpoint (e.g. days until quality rejection under a specified storage protocol). Record cultivar, harvest date/season, farm, maturity method, storage temperature/humidity, ethylene exposure, disease/decay, and quality measurements over time. Split evaluation by farm or harvest batch to avoid leakage. Monitor drift and prediction error after deployment. Never treat synthetic-data metrics as evidence of real-world accuracy.
