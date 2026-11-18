from pathlib import Path
import pandas as pd
from feature_engineering import add_features, TARGET

ROOT = Path(__file__).resolve().parents[1]

def build_feature_files():
    raw_dir = ROOT / "data" / "raw"
    feature_dir = ROOT / "data" / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    for version in (1, 2):
        source = raw_dir / f"mango_v{version}.csv"
        if not source.exists():
            raise FileNotFoundError(f"Missing {source}. Run src/generate_data.py first.")
        df = pd.read_csv(source)
        df = add_features(df)
        df = df.dropna(subset=[TARGET])
        df.to_csv(feature_dir / f"mango_features_v{version}.csv", index=False)
        print(f"Saved {len(df)} processed rows to mango_features_v{version}.csv")

if __name__ == "__main__":
    build_feature_files()
