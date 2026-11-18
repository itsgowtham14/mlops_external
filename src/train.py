from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from feature_engineering import FEATURES, NUMERIC_INPUTS, CATEGORICAL_INPUTS, TARGET

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"
EXPERIMENT = "mango-shelf-life"

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    frames = [pd.read_csv(FEATURE_DIR / f"mango_features_v{i}.csv") for i in (1, 2)]
    data = pd.concat(frames, ignore_index=True)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_INPUTS + ["heat_exposure_index", "handling_damage_index"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_INPUTS)
    ])
    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42
        )
    }

    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    results = []
    best = None

    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            metrics = {
                "MAE": float(mean_absolute_error(y_test, pred)),
                "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
                "R2": float(r2_score(y_test, pred))
            }
            mlflow.log_param("model_type", name)
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_metrics({k.lower(): v for k, v in metrics.items()})
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name="MangoShelfLifeModel",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )
            results.append({"model": name, **metrics, "run_id": run.info.run_id})
            if best is None or metrics["MAE"] < best["metrics"]["MAE"]:
                best = {"name": name, "pipeline": pipeline, "metrics": metrics, "run_id": run.info.run_id}

    joblib.dump(best["pipeline"], MODEL_DIR / "selected_model.joblib")
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(results, indent=2))
    (MODEL_DIR / "selected_model.txt").write_text(best["name"])
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSelected {best['name']} by lowest MAE.")
    print("Registered model: MangoShelfLifeModel (see MLflow UI for versions).")
    print("Saved API model to models/selected_model.joblib")

if __name__ == "__main__":
    main()
