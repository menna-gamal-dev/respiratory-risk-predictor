from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

print("Fetching Exasens from UCI...")
exasens = fetch_ucirepo(id=566)

df = exasens.data.original.copy()

# Save a local working copy for reproducibility.
output = DATA_DIR / "exasens.csv"
df.to_csv(output, index=False)

print(f"Saved: {output}")
print(f"Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst rows:")
print(df.head())
