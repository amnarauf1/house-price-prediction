"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HOUSE PRICE PREDICTION — ISLAMABAD                                          ║
║  AIC354 Machine Learning Fundamentals Lab | COMSATS University Islamabad     ║
║  Instructor: Mr. Anayat Ullah                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

  PROJECT STRUCTURE:
    house_price_prediction/
    ├── main.py              ← YOU ARE HERE  (run this)
    ├── scraper.py           ← Part 1: Scrapes Zameen.com
    ├── preprocessing.py     ← Part 2: Cleans data, engineers features
    ├── eda.py               ← Part 3: Generates 8 EDA charts
    ├── models.py            ← Parts 4&5: Trains 9 models, evaluates
    ├── predict.py           ← Part 6: Interactive prediction terminal
    ├── dataset/             ← CSVs saved here (auto-created)
    ├── models/              ← Trained model .pkl files (auto-created)
    └── visuals/             ← Charts (auto-created)

  USAGE:
    python main.py                   # full pipeline (scrape → train → predict)
    python main.py --skip-scrape     # skip scraping, reuse existing CSV
    python main.py --predict-only    # jump straight to prediction terminal

  INSTALL REQUIREMENTS:
    pip install requests beautifulsoup4 pandas numpy scikit-learn
                matplotlib seaborn joblib xgboost catboost lightgbm
"""

import os
import sys
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── Set up logging before importing submodules ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run.log", mode="w")
    ]
)
log = logging.getLogger(__name__)

# ── Import project modules ────────────────────────────────────────────────────
from scraper       import run_scraper
from preprocessing import run_preprocessing
from eda           import run_eda
from models        import run_training
from predict       import run_prediction

RAW_CSV = "dataset/real_estate_dataset.csv"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="House Price Prediction — Islamabad (COMSATS AIC354)"
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip scraping; reuse existing dataset/real_estate_dataset.csv"
    )
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Skip everything and jump straight to the prediction terminal"
    )
    args = parser.parse_args()

    print("\n╔" + "═"*64 + "╗")
    print("║  HOUSE PRICE PREDICTION — ISLAMABAD" + " "*28 + "║")
    print("║  AIC354 ML Lab | COMSATS University Islamabad" + " "*18 + "║")
    print("╚" + "═"*64 + "╝\n")

    # ── Predict-only shortcut ─────────────────────────────────────────────────
    if args.predict_only:
        run_prediction()
        return

    # ── Step 1: Scrape ────────────────────────────────────────────────────────
    if args.skip_scrape:
        if not os.path.exists(RAW_CSV):
            sys.exit(f"[ERROR] --skip-scrape used but '{RAW_CSV}' not found. "
                     f"Run without --skip-scrape first.")
        log.info(f"[Main] Skipping scraper. Using existing {RAW_CSV}")
    else:
        run_scraper()

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    run_preprocessing()

    # ── Step 3: EDA ───────────────────────────────────────────────────────────
    run_eda()

    # ── Steps 4 & 5: Train & Evaluate ─────────────────────────────────────────
    run_training()

    # ── Step 6: Interactive Prediction ────────────────────────────────────────
    run_prediction()

    print("\n╔" + "═"*64 + "╗")
    print("║  PIPELINE COMPLETE ✓" + " "*43 + "║")
    print("║  Outputs:" + " "*54 + "║")
    print("║    dataset/real_estate_dataset.csv  — raw scraped data" + " "*9 + "║")
    print("║    dataset/processed_dataset.csv    — cleaned data" + " "*13 + "║")
    print("║    dataset/model_results.csv        — model scores" + " "*13 + "║")
    print("║    models/house_price_model.pkl     — best model" + " "*14 + "║")
    print("║    visuals/01_*.png … 12_*.png      — all charts" + " "*14 + "║")
    print("║    run.log                          — full execution log" + " "*8 + "║")
    print("╚" + "═"*64 + "╝\n")


if __name__ == "__main__":
    main()
