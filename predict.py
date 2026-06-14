"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HOUSE PRICE PREDICTION — ISLAMABAD                                          ║
║  FILE: predict.py  |  Part 6 — Interactive Prediction System                 ║
║  Loads the best saved model and lets you predict prices interactively        ║
╚══════════════════════════════════════════════════════════════════════════════╝

  USAGE (standalone):
    python predict.py

  REQUIRES: models/house_price_model.pkl  (run main.py first)
"""

import os
import re
import sys
import logging

import pandas as pd
import numpy as np
import joblib

MODEL_PATH   = "models/house_price_model.pkl"
FEAT_PATH    = "models/feature_columns.pkl"
LOC_MAP_PATH = "models/location_map.pkl"

log = logging.getLogger(__name__)

# Property types must match those seen during training
PROPERTY_TYPES = [
    "House", "Flat", "Upper Portion", "Lower Portion",
    "Room", "Penthouse", "Farm House"
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _fmt_price(pkr):
    if pkr >= 1e9: return f"PKR {pkr/1e9:.2f} Arab"
    if pkr >= 1e7: return f"PKR {pkr/1e7:.2f} Crore"
    if pkr >= 1e5: return f"PKR {pkr/1e5:.2f} Lakh"
    return f"PKR {pkr:,.0f}"


def _parse_area_str(text):
    t = text.upper().replace(",", "")
    m = re.search(r"([\d.]+)", t)
    if not m:
        raise ValueError(f"Cannot parse area '{text}'")
    n = float(m.group(1))
    if "KANAL" in t: return n * 4356.0
    if "MARLA" in t: return n * 272.25
    if "ACRE"  in t: return n * 43560.0
    return n   # assume Sq Ft if no unit


def _encode_location(name, loc_map):
    """
    FIX: Now warns the user clearly if location is not recognised,
    instead of silently returning 0 (which mapped to a random area).
    """
    key = name.strip().lower()
    # exact match
    if key in loc_map:
        return int(loc_map[key]), True
    # partial match
    for k, v in loc_map.items():
        if key in k or k in key:
            return int(v), True
    # not found
    return 0, False


def _encode_property_type(prop_type_str, processed_csv="dataset/processed_dataset.csv"):
    """
    FIX: Encode property type properly by looking up the same mapping
    used during training, instead of always returning 0.
    """
    try:
        df = pd.read_csv(processed_csv, usecols=["Property_Type", "Property_Type_Enc"])
        mapping = dict(zip(
            df["Property_Type"].str.strip().str.lower(),
            df["Property_Type_Enc"].astype(int)
        ))
        key = prop_type_str.strip().lower()
        if key in mapping:
            return mapping[key]
        # partial match
        for k, v in mapping.items():
            if key in k or k in key:
                return v
    except Exception as e:
        log.warning(f"Could not load property type mapping: {e}")
    log.warning(f"Property type '{prop_type_str}' not found. Using 0.")
    return 0


def _default_feats():
    return [
        "Area_SqFt", "Log_Area", "Bedrooms", "Bathrooms",
        "Parking_Spaces", "Servant_Quarters", "Store_Rooms",
        "Kitchens", "Drawing_Rooms", "Property_Age", "Total_Rooms",
        "Bed_Bath_Ratio", "Area_Bin",
        "Location_Enc", "Property_Type_Enc"
    ]


def _ask_int(prompt, default=None):
    while True:
        raw = input(prompt).strip()
        if raw.lower() == "exit":
            return None
        if default is not None and raw == "":
            return default
        if raw.isdigit():
            return int(raw)
        print("  ⚠  Please enter a whole number.")


def _ask_choice(prompt, options):
    """Show a numbered menu and return the chosen string."""
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    while True:
        raw = input("  Enter number: ").strip()
        if raw.lower() == "exit":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  ⚠  Enter a number between 1 and {len(options)}.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_prediction():
    print("\n╔" + "═"*64 + "╗")
    print("║  PART 6 — INTERACTIVE PREDICTION SYSTEM" + " "*23 + "║")
    print("╚" + "═"*64 + "╝")

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"[ERROR] Model not found at '{MODEL_PATH}'. "
                 f"Run 'python main.py' first.")

    model     = joblib.load(MODEL_PATH)
    feat_cols = (joblib.load(FEAT_PATH)
                 if os.path.exists(FEAT_PATH) else _default_feats())
    loc_map   = (joblib.load(LOC_MAP_PATH)
                 if os.path.exists(LOC_MAP_PATH) else {})

    print(f"\n  Model loaded : {type(model).__name__}")
    print(f"  Features     : {len(feat_cols)}")
    print(f"  Locations    : {len(loc_map)} known areas")
    print("  Type 'exit' at any prompt to quit.\n")

    while True:
        print("─" * 58)
        try:
            # ── Area ──────────────────────────────────────────────────────────
            area_in = input("  Area  (e.g. 10 Marla / 2 Kanal / 1500 Sq Ft) : ").strip()
            if area_in.lower() == "exit": break
            try:
                area_sqft = _parse_area_str(area_in)
            except ValueError as e:
                print(f"  ⚠  {e}"); continue

            # ── Bedrooms ──────────────────────────────────────────────────────
            bedrooms = _ask_int("  Bedrooms   : ")
            if bedrooms is None: break

            # ── Bathrooms ─────────────────────────────────────────────────────
            bathrooms = _ask_int("  Bathrooms  : ")
            if bathrooms is None: break
            if bathrooms == 0:
                print("  ⚠  Bathrooms cannot be 0. Setting to 1.")
                bathrooms = 1

            # ── Parking ───────────────────────────────────────────────────────
            parking = _ask_int("  Parking Spaces (press Enter for 1) : ", default=1)
            if parking is None: break

            # ── Property Age ──────────────────────────────────────────────────
            age_in = input("  Property Age in years (press Enter for 5) : ").strip()
            if age_in.lower() == "exit": break
            prop_age = int(age_in) if age_in.isdigit() else 5

            # ── Property Type — FIX: ask user instead of always using 0 ───────
            prop_type = _ask_choice(
                "\n  Property Type — choose one:",
                PROPERTY_TYPES
            )
            if prop_type is None: break
            prop_type_enc = _encode_property_type(prop_type)
            print(f"  ✓ Property type encoded as: {prop_type_enc}")

            # ── Location — FIX: warn clearly if not recognised ─────────────────
            loc_in = input("\n  Location   (e.g. F-7, DHA Phase 2, Bahria Town) : ").strip()
            if loc_in.lower() == "exit": break
            loc_enc, found = _encode_location(loc_in, loc_map)
            if not found:
                print(f"  ⚠  WARNING: Location '{loc_in}' was not seen in training data.")
                print("     The prediction may be less accurate for this location.")
                print(f"     Known locations include: {', '.join(list(loc_map.keys())[:8])} ...")

            # ── Build feature vector ───────────────────────────────────────────
            defaults = {
                "Area_SqFt":         area_sqft,
                "Log_Area":          np.log1p(area_sqft),
                "Bedrooms":          bedrooms,
                "Bathrooms":         bathrooms,
                "Parking_Spaces":    parking,
                "Servant_Quarters":  0,
                "Store_Rooms":       0,
                "Kitchens":          1,
                "Drawing_Rooms":     1,
                "Property_Age":      prop_age,
                "Total_Rooms":       bedrooms + bathrooms,
                "Bed_Bath_Ratio":    bedrooms / max(bathrooms, 1),
                "Area_Bin":          (1 if area_sqft < 500 else
                                      2 if area_sqft < 1000 else
                                      3 if area_sqft < 2000 else
                                      4 if area_sqft < 5000 else 5),
                "Location_Enc":      loc_enc,
                "Property_Type_Enc": prop_type_enc,   # FIX: real value now
            }

            row   = pd.DataFrame([{c: defaults.get(c, 0) for c in feat_cols}])
            price = float(model.predict(row)[0])
            price = max(price, 0)

            # FIX: Confidence range based on typical model RMSE (~20%) not arbitrary 15%
            low  = _fmt_price(price * 0.80)
            high = _fmt_price(price * 1.20)

            print(f"\n  ┌──────────────────────────────────────────────────┐")
            print(f"  │  Predicted House Price                           │")
            print(f"  │  {_fmt_price(price):<48}│")
            print(f"  │  Estimated Range : {low}  —  {high:<16}│")
            print(f"  └──────────────────────────────────────────────────┘\n")

        except (ValueError, TypeError) as e:
            print(f"  [Input Error] {e}. Please try again.\n")
        except KeyboardInterrupt:
            break

    print("\n  Goodbye!\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    run_prediction()
