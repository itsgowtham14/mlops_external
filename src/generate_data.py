from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

def make_dataset(n: int = 300, version: int = 1, seed: int = 42) -> pd.DataFrame:
    """Create plausible synthetic shipment records; values are illustrative, not field measurements."""
    rng = np.random.default_rng(seed + version)
    maturity = rng.uniform(70, 100, n)                 # maturity index (0-100)
    pulp_temp = rng.normal(25 if version == 1 else 29, 2.8 if version == 1 else 3.2, n).clip(15, 40)
    brix = rng.normal(14.0, 1.8, n).clip(9, 21)
    bruising = rng.beta(1.5, 9, n) * 35
    transport = rng.gamma(2.2, 5.0 if version == 1 else 7.5, n).clip(1, 48)
    cooling = rng.gamma(1.8, 2.0 if version == 1 else 4.0, n).clip(0.2, 24)
    packaging = rng.choice(["ventilated_crate", "corrugated_carton", "plastic_crate"], n,
                           p=[0.35, 0.40, 0.25])

    # Engineered risk terms used to generate a noisy but learnable target.
    heat = np.maximum(pulp_temp - 18, 0) * transport / 24
    damage = bruising * 0.10 + cooling * 0.32
    packaging_penalty = pd.Series(packaging).map({
        "ventilated_crate": 0.0,
        "corrugated_carton": 0.25,
        "plastic_crate": 0.55
    }).to_numpy()

    shelf_life = (
        18.0
        - 0.105 * (maturity - 70)
        + 0.30 * (brix - 14)
        - 0.16 * bruising
        - 0.095 * transport
        - 0.22 * cooling
        - 0.13 * heat
        - 0.12 * damage
        - packaging_penalty
        + rng.normal(0, 1.15, n)
    )
    shelf_life = np.clip(shelf_life, 1.0, 24.0).round(1)

    return pd.DataFrame({
        "shipment_id": [f"V{version}-{i+1:04d}" for i in range(n)],
        "harvest_maturity_index": maturity.round(1),
        "pulp_temperature_c": pulp_temp.round(1),
        "sweetness_brix": brix.round(1),
        "bruising_percentage": bruising.round(1),
        "transport_duration_hours": transport.round(1),
        "cooling_delay_hours": cooling.round(1),
        "packaging_type": packaging,
        "remaining_shelf_life_days": shelf_life,
        "dataset_version": f"v{version}"
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=300)
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    v1 = make_dataset(args.rows, version=1, seed=42)
    v2 = make_dataset(args.rows, version=2, seed=42)
    v1.to_csv(RAW / "mango_v1.csv", index=False)
    v2.to_csv(RAW / "mango_v2.csv", index=False)
    print(f"Created V1: {len(v1)} rows; V2: {len(v2)} rows")
    print("V2 average transport/cooling:",
          round(v2.transport_duration_hours.mean(), 2),
          round(v2.cooling_delay_hours.mean(), 2))

if __name__ == "__main__":
    main()
