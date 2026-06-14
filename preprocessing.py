"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HOUSE PRICE PREDICTION — ISLAMABAD                                          ║
║  FILE: preprocessing.py  |  Part 2 — Data Preprocessing                     ║
║  Cleans raw CSV, engineers features, encodes, splits, saves                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

  USAGE (standalone):
    python preprocessing.py

  INPUT:  dataset/real_estate_dataset.csv
  OUTPUT: dataset/processed_dataset.csv
          dataset/X_train.csv  /  X_test.csv  /  y_train.csv  /  y_test.csv
          models/feature_columns.pkl
          models/location_map.pkl
"""

import os
import re
import sys
import logging

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder

RAW_CSV       = "dataset/real_estate_dataset.csv"
PROCESSED_CSV = "dataset/processed_dataset.csv"
FEAT_PATH     = "models/feature_columns.pkl"
LOC_MAP_PATH  = "models/location_map.pkl"
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

os.makedirs("dataset", exist_ok=True)
os.makedirs("models",  exist_ok=True)

log = logging.getLogger(__name__)


# ── PARSERS ───────────────────────────────────────────────────────────────────

def _parse_price(v):
    if pd.isna(v): return None
    t = str(v).upper().replace(",", "")
    m = re.search(r"([\d.]+)", t)
    if not m: return None
    n = float(m.group(1))
    if "CRORE" in t or t.endswith("CR"): return n * 1e7
    if "LAKH"  in t or "LAC" in t:       return n * 1e5
    if "ARAB"  in t:                      return n * 1e9
    return n


def _parse_area(v):
    if pd.isna(v): return None
    t = str(v).upper().replace(",", "")
    m = re.search(r"([\d.]+)", t)
    if not m: return None
    n = float(m.group(1))
    if "KANAL" in t: return n * 4356.0
    if "MARLA" in t: return n * 272.25
    if "ACRE"  in t: return n * 43560.0
    return n


def _to_num(v, default=np.nan):
    if pd.isna(v): return default
    m = re.search(r"[\d.]+", str(v))
    return float(m.group()) if m else default


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_preprocessing():
    print("\n╔" + "═"*64 + "╗")
    print("║  PART 2 — DATA PREPROCESSING" + " "*34 + "║")
    print("╚" + "═"*64 + "╝")

    if not os.path.exists(RAW_CSV):
        sys.exit(f"[ERROR] Raw dataset not found at '{RAW_CSV}'. Run scraper first.")

    df = pd.read_csv(RAW_CSV)
    log.info(f"[Prep] Loaded {len(df)} rows × {len(df.columns)} columns")

    # Deduplication
    before = len(df)
    df = df.drop_duplicates(subset=["URL"]).drop_duplicates()
    log.info(f"[Prep] Removed {before - len(df)} duplicates → {len(df)} rows")

    # Parse Price & Area
    df["Price_PKR"] = df["Price"].apply(_parse_price)
    df["Area_SqFt"] = df["Area"].apply(_parse_area)

    # Numeric rooms
    for col in ["Bedrooms","Bathrooms","Built_in_Year","Parking_Spaces",
                "Servant_Quarters","Store_Rooms","Kitchens","Drawing_Rooms"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_num)

    # String columns
    for col in ["City","Property_Type","Location"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Drop rows missing target or key feature
    df = df.dropna(subset=["Price_PKR","Area_SqFt"])
    df = df[(df["Price_PKR"] >= 100_000) & (df["Price_PKR"] <= 5_000_000_000)]
    log.info(f"[Prep] After outlier removal: {len(df)} rows")

    # Impute missing numerics with median
    num_cols = ["Bedrooms","Bathrooms","Built_in_Year","Parking_Spaces",
                "Servant_Quarters","Store_Rooms","Kitchens","Drawing_Rooms"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Impute missing categoricals with mode
    for col in ["Property_Type","Location"]:
        if col in df.columns and df[col].notna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Feature engineering
    if "Built_in_Year" in df.columns:
        df["Property_Age"] = (2026 - df["Built_in_Year"]).clip(0, 100)
    else:
        df["Property_Age"] = 5

    df["Total_Rooms"]    = df["Bedrooms"].fillna(0) + df["Bathrooms"].fillna(0)
    df["Price_per_SqFt"] = df["Price_PKR"] / df["Area_SqFt"]

    # ── NEW: extra engineered features for better model accuracy ──────────────
    # Bed-to-Bath ratio (quality signal)
    df["Bed_Bath_Ratio"] = (df["Bedrooms"] / df["Bathrooms"].replace(0, 1)).clip(0, 10)
    # Log-transform area (reduces skew)
    df["Log_Area"]       = np.log1p(df["Area_SqFt"])
    # Price bracket bins (helps tree models)
    df["Area_Bin"]       = pd.cut(df["Area_SqFt"],
                                  bins=[0, 500, 1000, 2000, 5000, 1e9],
                                  labels=[1, 2, 3, 4, 5]).astype(float)

    # Label encoding
    le      = LabelEncoder()
    loc_map = {}
    for col in ["Location","Property_Type"]:
        if col in df.columns:
            df[col+"_Enc"] = le.fit_transform(df[col].astype(str))
            if col == "Location":
                loc_map = dict(zip(df[col].str.lower(), df[col+"_Enc"].astype(int)))

    joblib.dump(loc_map, LOC_MAP_PATH)

    # Save processed CSV
    df.to_csv(PROCESSED_CSV, index=False)
    log.info(f"[Prep] Processed dataset saved → {PROCESSED_CSV}")

    # Feature selection — includes new features
    FEAT_COLS = [c for c in [
        "Area_SqFt", "Log_Area", "Bedrooms", "Bathrooms",
        "Parking_Spaces", "Servant_Quarters", "Store_Rooms",
        "Kitchens", "Drawing_Rooms", "Property_Age", "Total_Rooms",
        "Bed_Bath_Ratio", "Area_Bin",
        "Location_Enc", "Property_Type_Enc"
    ] if c in df.columns]

    X = df[FEAT_COLS]
    y = df["Price_PKR"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log.info(f"[Prep] Train: {len(X_train)}  Test: {len(X_test)}")

    X_train.to_csv("dataset/X_train.csv", index=False)
    X_test.to_csv( "dataset/X_test.csv",  index=False)
    y_train.to_csv("dataset/y_train.csv", index=False)
    y_test.to_csv( "dataset/y_test.csv",  index=False)
    joblib.dump(FEAT_COLS, FEAT_PATH)

    log.info("[Prep] Done.")
    return X_train, X_test, y_train, y_test, FEAT_COLS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    run_preprocessing()
