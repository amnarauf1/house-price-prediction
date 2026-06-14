"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HOUSE PRICE PREDICTION — ISLAMABAD                                          ║
║  FILE: models.py  |  Parts 4 & 5 — Model Training & Evaluation              ║
║  Trains 9 regression models, evaluates, saves best                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

  MODELS TRAINED:
    1.  Linear Regression          (baseline)
    2.  Ridge Regression           (regularised linear)
    3.  Lasso Regression           (feature-selecting linear)
    4.  Decision Tree              (simple tree)
    5.  Random Forest              (200 trees, tuned)
    6.  Gradient Boosting          (sklearn, tuned)
    7.  XGBoost                    (if installed)
    8.  CatBoost                   (if installed)
    9.  Voting Ensemble            (RF + GB + XGB combined)

  USAGE (standalone):
    python models.py

  INPUT:  dataset/X_train.csv  X_test.csv  y_train.csv  y_test.csv
  OUTPUT: models/house_price_model.pkl
          dataset/model_results.csv
          visuals/09_model_comparison.png
          visuals/10_feature_importance.png
          visuals/11_actual_vs_predicted.png
          visuals/12_residuals.png
"""

import os
import sys
import logging

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model  import LinearRegression, Ridge, Lasso
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import (RandomForestRegressor,
                                   GradientBoostingRegressor,
                                   VotingRegressor)
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost  import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

MODEL_PATH  = "models/house_price_model.pkl"
FEAT_PATH   = "models/feature_columns.pkl"
PALETTE     = "viridis"
RANDOM_STATE = 42

os.makedirs("models",  exist_ok=True)
os.makedirs("visuals", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

sns.set_theme(style="whitegrid")
log = logging.getLogger(__name__)


# ── BUILD ALL MODELS ──────────────────────────────────────────────────────────

def _build_models():
    """
    Returns dict of model_name → estimator.
    All tree models are tuned for better accuracy on real-estate data.
    Linear models are wrapped in a StandardScaler pipeline.
    """
    models = {}

    # ── Linear models (need scaling) ──────────────────────────────────────────
    models["Linear Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression())
    ])
    models["Ridge Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=10.0))
    ])
    models["Lasso Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Lasso(alpha=100.0, max_iter=5000))
    ])

    # ── Tree models ───────────────────────────────────────────────────────────
    models["Decision Tree"] = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    models["Gradient Boosting"] = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )

    # ── Optional boosting libraries ───────────────────────────────────────────
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1
        )
    else:
        log.warning("[Train] xgboost not installed — skipping XGBoost.")

    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_seed=RANDOM_STATE,
            verbose=0
        )
    else:
        log.warning("[Train] catboost not installed — skipping CatBoost.")

    if HAS_LGB:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
    else:
        log.warning("[Train] lightgbm not installed — skipping LightGBM.")

    return models


def _build_ensemble(trained):
    """Build a VotingRegressor from the best available tree models."""
    candidates = []
    for name in ["Random Forest", "Gradient Boosting", "XGBoost", "CatBoost", "LightGBM"]:
        if name in trained:
            candidates.append((name.lower().replace(" ", "_"), trained[name]))
        if len(candidates) == 3:
            break
    if len(candidates) >= 2:
        return VotingRegressor(estimators=candidates, n_jobs=-1)
    return None


# ── EVALUATION HELPERS ────────────────────────────────────────────────────────

def _evaluate(model, X_test, y_test):
    yp   = model.predict(X_test)
    mae  = mean_absolute_error(y_test, yp)
    mse  = mean_squared_error(y_test, yp)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, yp)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}, yp


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def _plot_comparison(df_res):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")
    clrs = sns.color_palette(PALETTE, len(df_res))
    for ax, metric in zip(axes.flatten(), ["MAE", "MSE", "RMSE", "R2"]):
        vals = df_res[metric]
        bars = ax.bar(vals.index, vals.values, color=clrs, edgecolor="white")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:,.2f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig("visuals/09_model_comparison.png", dpi=150); plt.close()
    log.info("[Train] Saved: visuals/09_model_comparison.png")


def _plot_feature_importance(trained, X_train):
    for name in ["CatBoost", "LightGBM", "XGBoost",
                 "Gradient Boosting", "Random Forest", "Decision Tree"]:
        mdl = trained.get(name)
        if mdl is None:
            continue
        # unwrap Pipeline if needed
        if hasattr(mdl, "named_steps"):
            mdl = mdl.named_steps.get("model", mdl)
        if hasattr(mdl, "feature_importances_"):
            fi = pd.Series(mdl.feature_importances_,
                           index=X_train.columns).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            fi.plot(kind="barh", ax=ax, color="#3498DB")
            ax.set_title(f"Feature Importance — {name}",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig("visuals/10_feature_importance.png", dpi=150); plt.close()
            log.info("[Train] Saved: visuals/10_feature_importance.png")
            break


def _plot_actual_vs_predicted(y_test, y_pred, best_name):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test / 1e6, y_pred / 1e6,
               alpha=0.5, color="#2ECC71", edgecolors="w", s=40)
    lims = [min(y_test.min(), y_pred.min()) / 1e6,
            max(y_test.max(), y_pred.max()) / 1e6]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Price (PKR Millions)")
    ax.set_ylabel("Predicted Price (PKR Millions)")
    ax.set_title(f"Actual vs Predicted — {best_name}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("visuals/11_actual_vs_predicted.png", dpi=150); plt.close()
    log.info("[Train] Saved: visuals/11_actual_vs_predicted.png")


def _plot_residuals(y_test, y_pred, best_name):
    residuals = y_test.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred / 1e6, residuals / 1e6,
                    alpha=0.4, color="#E74C3C", edgecolors="w", s=35)
    axes[0].axhline(0, color="black", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("Predicted Price (PKR Millions)")
    axes[0].set_ylabel("Residual (PKR Millions)")
    axes[0].set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")

    sns.histplot(residuals / 1e6, ax=axes[1], kde=True, color="#E74C3C")
    axes[1].axvline(0, color="black", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Residual (PKR Millions)")
    axes[1].set_title("Residual Distribution", fontsize=12, fontweight="bold")

    fig.suptitle(f"Residual Analysis — {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("visuals/12_residuals.png", dpi=150); plt.close()
    log.info("[Train] Saved: visuals/12_residuals.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_training():
    print("\n╔" + "═"*64 + "╗")
    print("║  PARTS 4 & 5 — MODEL TRAINING & EVALUATION" + " "*20 + "║")
    print("╚" + "═"*64 + "╝")

    for f in ["dataset/X_train.csv", "dataset/X_test.csv",
              "dataset/y_train.csv", "dataset/y_test.csv"]:
        if not os.path.exists(f):
            sys.exit(f"[ERROR] '{f}' not found. Run preprocessing first.")

    X_train = pd.read_csv("dataset/X_train.csv")
    X_test  = pd.read_csv("dataset/X_test.csv")
    y_train = pd.read_csv("dataset/y_train.csv").squeeze()
    y_test  = pd.read_csv("dataset/y_test.csv").squeeze()

    log.info(f"[Train] X_train: {X_train.shape}  X_test: {X_test.shape}")
    log.info(f"[Train] Features: {list(X_train.columns)}")

    models  = _build_models()
    results = []
    trained = {}

    # ── Train all individual models ───────────────────────────────────────────
    for name, mdl in models.items():
        log.info(f"[Train] Training: {name} …")
        try:
            mdl.fit(X_train, y_train)
            metrics, _ = _evaluate(mdl, X_test, y_test)
            results.append({"Model": name, **metrics})
            trained[name] = mdl
            log.info(f"         R² = {metrics['R2']:.4f}  |  "
                     f"RMSE = PKR {metrics['RMSE']:,.0f}")
        except Exception as e:
            log.error(f"[Train] {name} failed: {e}")

    # ── Voting Ensemble ───────────────────────────────────────────────────────
    ensemble = _build_ensemble(trained)
    if ensemble:
        log.info("[Train] Training: Voting Ensemble …")
        try:
            ensemble.fit(X_train, y_train)
            metrics, _ = _evaluate(ensemble, X_test, y_test)
            results.append({"Model": "Voting Ensemble", **metrics})
            trained["Voting Ensemble"] = ensemble
            log.info(f"         R² = {metrics['R2']:.4f}  |  "
                     f"RMSE = PKR {metrics['RMSE']:,.0f}")
        except Exception as e:
            log.error(f"[Train] Voting Ensemble failed: {e}")

    # ── Results table ─────────────────────────────────────────────────────────
    df_res = pd.DataFrame(results).set_index("Model")

    print("\n" + "═" * 78)
    print("  MODEL PERFORMANCE COMPARISON")
    print("═" * 78)
    print(f"  {'Model':<22} {'MAE (PKR)':>14} {'RMSE (PKR)':>14} {'R²':>8}")
    print("  " + "─" * 65)
    for name, row in df_res.iterrows():
        marker = " ← BEST" if name == df_res["R2"].idxmax() else ""
        print(f"  {name:<22} {row['MAE']:>14,.0f} "
              f"{row['RMSE']:>14,.0f} {row['R2']:>8.4f}{marker}")
    print("═" * 78 + "\n")

    df_res.to_csv("dataset/model_results.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_comparison(df_res)
    _plot_feature_importance(trained, X_train)

    best_name = df_res["R2"].idxmax()
    _, best_preds = _evaluate(trained[best_name], X_test, y_test)
    _plot_actual_vs_predicted(y_test, best_preds, best_name)
    _plot_residuals(y_test, best_preds, best_name)

    # ── Save best model ───────────────────────────────────────────────────────
    joblib.dump(trained[best_name], MODEL_PATH)
    log.info(f"[Train] Best model '{best_name}' saved → {MODEL_PATH}")

    print(f"  ✓ Best model : {best_name}  "
          f"(R² = {df_res.loc[best_name, 'R2']:.4f})")
    print(f"  ✓ Saved to   : {MODEL_PATH}\n")

    return trained, df_res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    run_training()
