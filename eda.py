"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HOUSE PRICE PREDICTION — ISLAMABAD                                          ║
║  FILE: eda.py  |  Part 3 — Exploratory Data Analysis                        ║
║  Generates 8 visualisation charts from the processed dataset                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

  USAGE (standalone):
    python eda.py

  INPUT:  dataset/processed_dataset.csv
  OUTPUT: visuals/01_price_distribution.png  … visuals/08_*
"""

import os
import logging

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_CSV = "dataset/processed_dataset.csv"
PALETTE       = "viridis"

os.makedirs("visuals", exist_ok=True)
sns.set_theme(style="whitegrid")

log = logging.getLogger(__name__)


def run_eda():
    print("\n╔" + "═"*64 + "╗")
    print("║  PART 3 — EXPLORATORY DATA ANALYSIS" + " "*27 + "║")
    print("╚" + "═"*64 + "╝")

    if not os.path.exists(PROCESSED_CSV):
        log.warning("[EDA] Processed CSV not found. Skipping EDA.")
        return

    df = pd.read_csv(PROCESSED_CSV)

    # 1. Price distribution ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["Price_PKR"],           ax=axes[0], kde=True, color="#3498DB")
    sns.histplot(np.log1p(df["Price_PKR"]), ax=axes[1], kde=True, color="#E67E22")
    axes[0].set_title("Price Distribution (PKR)",  fontsize=13, fontweight="bold")
    axes[1].set_title("Log-Price Distribution",    fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Price (PKR)")
    axes[1].set_xlabel("log(Price+1)")
    plt.tight_layout()
    plt.savefig("visuals/01_price_distribution.png", dpi=150); plt.close()
    log.info("[EDA] Saved: visuals/01_price_distribution.png")

    # 2. Area vs Price scatter ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df["Area_SqFt"], df["Price_PKR"] / 1e6,
                    c=df["Bedrooms"].fillna(3), cmap=PALETTE,
                    alpha=0.6, edgecolors="w", s=55)
    plt.colorbar(sc, ax=ax, label="Bedrooms")
    ax.set_xlabel("Area (Sq Ft)")
    ax.set_ylabel("Price (PKR Millions)")
    ax.set_title("Area vs Price — coloured by Bedrooms", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("visuals/02_area_vs_price.png", dpi=150); plt.close()
    log.info("[EDA] Saved: visuals/02_area_vs_price.png")

    # 3. Bedroom analysis ──────────────────────────────────────────────────────
    if "Bedrooms" in df.columns:
        bp = df.groupby("Bedrooms")["Price_PKR"].median() / 1e6
        fig, ax = plt.subplots(figsize=(10, 5))
        bp.plot(kind="bar", ax=ax, color="#3498DB", edgecolor="white")
        ax.set_xlabel("Bedrooms")
        ax.set_ylabel("Median Price (PKR Millions)")
        ax.set_title("Median Price by Bedrooms", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        plt.savefig("visuals/03_bedroom_analysis.png", dpi=150); plt.close()
        log.info("[EDA] Saved: visuals/03_bedroom_analysis.png")

    # 4. Correlation heatmap ───────────────────────────────────────────────────
    keep = [c for c in [
        "Price_PKR", "Area_SqFt", "Bedrooms", "Bathrooms",
        "Parking_Spaces", "Total_Rooms", "Property_Age",
        "Price_per_SqFt", "Bed_Bath_Ratio"
    ] if c in df.columns]
    if keep:
        corr = df[keep].corr()
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(corr,
                    mask=np.triu(np.ones_like(corr, dtype=bool)),
                    annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, square=True, linewidths=0.5)
        ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("visuals/04_correlation_heatmap.png", dpi=150); plt.close()
        log.info("[EDA] Saved: visuals/04_correlation_heatmap.png")

    # 5. Top locations ─────────────────────────────────────────────────────────
    if "Location" in df.columns:
        top = (df.groupby("Location")["Price_PKR"]
                  .median()
                  .sort_values(ascending=False)
                  .head(15) / 1e6)
        fig, ax = plt.subplots(figsize=(12, 6))
        top.plot(kind="barh", ax=ax, color=sns.color_palette(PALETTE, len(top)))
        ax.set_xlabel("Median Price (PKR Millions)")
        ax.set_title("Top 15 Locations by Median Price", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig("visuals/05_location_prices.png", dpi=150); plt.close()
        log.info("[EDA] Saved: visuals/05_location_prices.png")

    # 6. Property type distribution ────────────────────────────────────────────
    if "Property_Type" in df.columns:
        pt = df["Property_Type"].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(10, 5))
        pt.plot(kind="bar", ax=ax,
                color=sns.color_palette(PALETTE, len(pt)), edgecolor="white")
        ax.set_xlabel("Property Type")
        ax.set_ylabel("Count")
        ax.set_title("Property Type Distribution", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig("visuals/06_property_types.png", dpi=150); plt.close()
        log.info("[EDA] Saved: visuals/06_property_types.png")

    # 7. Price per SqFt by location (NEW) ──────────────────────────────────────
    if "Price_per_SqFt" in df.columns and "Location" in df.columns:
        pps = (df.groupby("Location")["Price_per_SqFt"]
                  .median()
                  .sort_values(ascending=False)
                  .head(15))
        fig, ax = plt.subplots(figsize=(12, 6))
        pps.plot(kind="barh", ax=ax, color=sns.color_palette("magma", len(pps)))
        ax.set_xlabel("Median Price per Sq Ft (PKR)")
        ax.set_title("Price per Sq Ft — Top 15 Locations", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig("visuals/07_price_per_sqft.png", dpi=150); plt.close()
        log.info("[EDA] Saved: visuals/07_price_per_sqft.png")

    # 8. Property Age vs Price ─────────────────────────────────────────────────
    if "Property_Age" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(df["Property_Age"], df["Price_PKR"] / 1e6,
                   alpha=0.4, color="#9B59B6", edgecolors="w", s=40)
        ax.set_xlabel("Property Age (Years)")
        ax.set_ylabel("Price (PKR Millions)")
        ax.set_title("Property Age vs Price", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("visuals/08_age_vs_price.png", dpi=150); plt.close()
        log.info("[EDA] Saved: visuals/08_age_vs_price.png")

    log.info("[EDA] All charts saved to visuals/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    run_eda()
