from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
raw_path = BASE_DIR / "data" / "raw" / "medical_appointment.csv"
feature_path = BASE_DIR / "data" / "features" / "engineered_features.csv"

print("-> RAW DATA <-")
raw_df = pd.read_csv(raw_path)
print(f"Shape: {raw_df.shape}")
print(f"Columns: {raw_df.columns.tolist()}")
print(raw_df.head())

print("\n-> FEATURE DATA <-")
feature_df = pd.read_csv(feature_path)
print(f"Shape: {feature_df.shape}")
print(f"Columns: {feature_df.columns.tolist()}")
print(feature_df.head())
