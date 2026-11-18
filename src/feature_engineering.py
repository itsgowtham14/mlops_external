import pandas as pd

TARGET = "remaining_shelf_life_days"
NUMERIC_INPUTS = [
    "harvest_maturity_index", "pulp_temperature_c", "sweetness_brix",
    "bruising_percentage", "transport_duration_hours", "cooling_delay_hours"
]
CATEGORICAL_INPUTS = ["packaging_type"]
FEATURES = NUMERIC_INPUTS + CATEGORICAL_INPUTS + [
    "heat_exposure_index", "handling_damage_index"
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Temperature above 18°C, weighted by journey duration in days.
    out["heat_exposure_index"] = (
        (out["pulp_temperature_c"] - 18).clip(lower=0)
        * out["transport_duration_hours"] / 24
    )
    # Combined handling stress proxy; higher means greater risk.
    out["handling_damage_index"] = (
        out["bruising_percentage"] * 0.10
        + out["cooling_delay_hours"] * 0.32
    )
    return out
