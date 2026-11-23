"""Reproducible training, evaluation, visualization, and inference pipeline
for the HouseValue-ML California Housing project (ITE 351 Stage A).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "MedHouseVal"

ROOT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ROOT_DIR / "assets" / "figures"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"


def compute_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute descriptive statistics, missing counts, cap statistics, and correlations."""
    desc = df.describe().T
    stats = {}
    for col in df.columns:
        row = desc.loc[col]
        stats[col] = {
            "mean": round(float(row["mean"]), 4),
            "std": round(float(row["std"]), 4),
            "min": round(float(row["min"]), 4),
            "q25": round(float(row["25%"]), 4),
            "median": round(float(row["50%"]), 4),
            "q75": round(float(row["75%"]), 4),
            "max": round(float(row["max"]), 4),
        }

    capped_mask = df[TARGET_COL] >= 5.0
    capped_count = int(capped_mask.sum())
    capped_pct = round(float(capped_count / len(df) * 100.0), 2)

    corr_series = df.corr(numeric_only=True)[TARGET_COL].drop(TARGET_COL)
    correlations = {
        feat: round(float(val), 4)
        for feat, val in corr_series.sort_values(ascending=False).items()
    }

    missing_per_col = {col: int(cnt) for col, cnt in df.isna().sum().items()}

    return {
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "feature_names": [c for c in df.columns if c != TARGET_COL],
        "target_name": TARGET_COL,
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_per_column": missing_per_col,
        "capped_at_500k_count": capped_count,
        "capped_at_500k_pct": capped_pct,
        "descriptive_stats": stats,
        "target_correlations": correlations,
    }


def build_models() -> dict[str, Any]:
    """Construct the three candidate scikit-learn regression models."""
    return {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    """Train all candidate models, run 5-fold CV, and evaluate on the held-out test set."""
    models = build_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    metrics_list: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, estimator in models.items():
        t0 = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        y_pred = estimator.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        cv_res = cross_validate(
            build_models()[name],
            X_train,
            y_train,
            cv=cv,
            scoring={"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"},
            n_jobs=-1,
        )
        cv_r2_mean = float(np.mean(cv_res["test_r2"]))
        cv_r2_std = float(np.std(cv_res["test_r2"]))
        cv_rmse_mean = float(-np.mean(cv_res["test_neg_rmse"]))
        cv_rmse_std = float(np.std(-cv_res["test_neg_rmse"]))

        metrics_list.append(
            {
                "model": name,
                "test_mae": round(mae, 4),
                "test_mse": round(mse, 4),
                "test_rmse": round(rmse, 4),
                "test_r2": round(r2, 4),
                "test_mae_usd": round(mae * 100_000.0, 0),
                "test_rmse_usd": round(rmse * 100_000.0, 0),
                "cv_r2_mean": round(cv_r2_mean, 4),
                "cv_r2_std": round(cv_r2_std, 4),
                "cv_rmse_mean": round(cv_rmse_mean, 4),
                "cv_rmse_std": round(cv_rmse_std, 4),
                "fit_time_sec": round(fit_seconds, 3),
            }
        )
        fitted_models[name] = estimator
        predictions[name] = y_pred

    return metrics_list, fitted_models, predictions


def predict_house_value(model: Any, sample_features: dict[str, float]) -> float:
    """Predict median house value in USD for a single district feature dictionary."""
    ordered_cols = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    sample_df = pd.DataFrame([[sample_features[c] for c in ordered_cols]], columns=ordered_cols)
    pred_100k = float(model.predict(sample_df)[0])
    return round(pred_100k * 100_000.0, 2)


def generate_figures(
    df: pd.DataFrame,
    y_test: pd.Series,
    metrics_list: list[dict[str, Any]],
    best_model_name: str,
    best_pred: np.ndarray,
    perm_importance_df: pd.DataFrame,
    dataset_summary: dict[str, Any],
) -> None:
    """Generate and save all four publication-quality EDA and evaluation figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.9,
            "font.size": 10.5,
        }
    )

    # 1. Geographic scatter map
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=200)
    scatter = ax.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df[TARGET_COL],
        s=np.clip(df["Population"] / 70.0, 6.0, 90.0),
        cmap="viridis",
        alpha=0.48,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, shrink=0.92)
    cbar.set_label("Median House Value (100,000 USD)", fontsize=10.5, fontweight="bold")

    hubs = [
        ("San Francisco Bay Area", (-122.42, 37.77), (-124.25, 36.75)),
        ("Los Angeles Basin", (-118.24, 34.05), (-121.60, 33.40)),
        ("San Diego", (-117.16, 32.72), (-120.20, 32.15)),
        ("Central Valley", (-119.78, 36.75), (-117.40, 37.95)),
    ]
    for label, target_xy, text_xy in hubs:
        ax.annotate(
            label,
            xy=target_xy,
            xytext=text_xy,
            fontsize=9.5,
            fontweight="bold",
            color="#1F2937",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#6B7280", "alpha": 0.92},
            arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.2},
        )

    ax.set_title(
        "Spatial Distribution of California Median House Values (1990 U.S. Census, n = 20,640)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Longitude (Degrees West)", fontsize=11)
    ax.set_ylabel("Latitude (Degrees North)", fontsize=11)
    ax.set_xlim(-124.8, -113.8)
    ax.set_ylim(31.8, 42.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_california_housing_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Target distribution and correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=200)

    sns.histplot(
        df[TARGET_COL],
        bins=45,
        kde=True,
        color="#2563EB",
        alpha=0.65,
        ax=axes[0],
        line_kws={"linewidth": 2.0, "color": "#1E3A8A"},
    )
    axes[0].axvline(5.0, color="#DC2626", linestyle="--", linewidth=2.0)
    capped_cnt = dataset_summary["capped_at_500k_count"]
    capped_pct = dataset_summary["capped_at_500k_pct"]
    axes[0].annotate(
        f"Census Cap at 5.0 (500k USD)\n{capped_cnt} blocks ({capped_pct:.2f}%)",
        xy=(5.0, axes[0].get_ylim()[1] * 0.72),
        xytext=(3.28, axes[0].get_ylim()[1] * 0.80),
        fontsize=9.5,
        fontweight="bold",
        color="#991B1B",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#FEF2F2", "ec": "#DC2626"},
        arrowprops={"arrowstyle": "->", "color": "#DC2626", "lw": 1.3},
    )
    axes[0].set_title("(a) Target Distribution (MedHouseVal)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Median House Value (100,000 USD)", fontsize=10.5)
    axes[0].set_ylabel("Number of Block Groups", fontsize=10.5)

    corr_mat = df.corr(numeric_only=True)
    sns.heatmap(
        corr_mat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8.5},
        cbar_kws={"shrink": 0.82},
        ax=axes[1],
    )
    axes[1].set_title("(b) Feature & Target Correlation Matrix", fontsize=12, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].tick_params(axis="y", rotation=0)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "02_target_distribution_and_correlations.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

    # 3. Model comparison bar charts
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=200)
    models_df = pd.DataFrame(metrics_list)
    x = np.arange(len(models_df))
    width = 0.35

    bars_rmse = axes[0].bar(
        x - width / 2, models_df["test_rmse"], width, label="Test RMSE", color="#1D4ED8"
    )
    bars_mae = axes[0].bar(
        x + width / 2, models_df["test_mae"], width, label="Test MAE", color="#60A5FA"
    )
    for bar in list(bars_rmse) + list(bars_mae):
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.015,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models_df["model"], fontsize=10)
    axes[0].set_ylim(0, max(models_df["test_rmse"]) * 1.18)
    axes[0].set_ylabel("Error (100,000 USD) — Lower is Better", fontsize=10.5)
    axes[0].set_title("(a) Prediction Error Comparison (RMSE & MAE)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper right")

    bars_r2 = axes[1].bar(
        x - width / 2, models_df["test_r2"], width, label="Test R²", color="#047857"
    )
    bars_cv = axes[1].bar(
        x + width / 2,
        models_df["cv_r2_mean"],
        width,
        yerr=models_df["cv_r2_std"],
        capsize=4,
        label="5-Fold CV R² (mean ± std)",
        color="#34D399",
    )
    for bar in list(bars_r2) + list(bars_cv):
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models_df["model"], fontsize=10)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Coefficient of Determination (R²) — Higher is Better", fontsize=10.5)
    axes[1].set_title("(b) Explained Variance Comparison (Test vs. 5-Fold CV)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Predicted vs. Actual + Permutation Importance
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5), dpi=200)

    axes[0].scatter(
        y_test,
        best_pred,
        s=14,
        alpha=0.35,
        color="#2563EB",
        edgecolors="none",
        label="Test Districts (n = 4,128)",
    )
    axes[0].plot([0.1, 5.15], [0.1, 5.15], "r--", lw=1.8, label="Ideal Prediction (y = x)")
    axes[0].axvline(5.0, color="#D97706", linestyle=":", lw=1.5, label="Census Cap (5.0)")

    best_row = next(m for m in metrics_list if m["model"] == best_model_name)
    axes[0].text(
        0.05,
        0.88,
        f"{best_model_name}\nTest RMSE = {best_row['test_rmse']:.4f}\nTest R² = {best_row['test_r2']:.4f}",
        transform=axes[0].transAxes,
        fontsize=9.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#9CA3AF", "alpha": 0.92},
    )
    axes[0].set_title(
        f"(a) Predicted vs. Actual Values ({best_model_name})",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Actual Median House Value (100,000 USD)", fontsize=10.5)
    axes[0].set_ylabel("Predicted Median House Value (100,000 USD)", fontsize=10.5)
    axes[0].set_xlim(0.0, 5.3)
    axes[0].set_ylim(0.0, 5.3)
    axes[0].legend(loc="lower right", fontsize=9)

    perm_sorted = perm_importance_df.sort_values("importance_mean", ascending=True)
    y_pos = np.arange(len(perm_sorted))
    bars = axes[1].barh(
        y_pos,
        perm_sorted["importance_mean"],
        xerr=perm_sorted["importance_std"],
        capsize=3.5,
        color="#0F766E",
        alpha=0.88,
    )
    max_imp = float(perm_sorted["importance_mean"].max())
    for bar, val in zip(bars, perm_sorted["importance_mean"]):
        axes[1].text(
            val + max_imp * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(perm_sorted["feature"], fontsize=10)
    axes[1].set_xlim(0, max_imp * 1.18)
    axes[1].set_xlabel("Mean R² Decrease on Test Set (10 Repeats)", fontsize=10.5)
    axes[1].set_title(
        f"(b) Permutation Feature Importance ({best_model_name})",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "04_predicted_vs_actual_and_importance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    """Execute the end-to-end training, evaluation, plotting, and inference pipeline."""
    dataset = fetch_california_housing(as_frame=True)
    df: pd.DataFrame = dataset.frame.copy()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    dataset_summary = compute_dataset_summary(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    metrics_list, fitted_models, predictions = evaluate_models(
        X_train, X_test, y_train, y_test
    )

    best_metric = max(metrics_list, key=lambda m: m["test_r2"])
    best_model_name = best_metric["model"]
    best_model = fitted_models[best_model_name]
    best_pred = predictions[best_model_name]

    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="r2",
    )
    perm_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    sample_profiles = [
        {
            "profile": "Coastal Bay Area District",
            "features": {
                "MedInc": 7.80,
                "HouseAge": 32.0,
                "AveRooms": 6.40,
                "AveBedrms": 1.08,
                "Population": 1150.0,
                "AveOccup": 2.60,
                "Latitude": 37.75,
                "Longitude": -122.42,
            },
        },
        {
            "profile": "Inland Central Valley District",
            "features": {
                "MedInc": 3.10,
                "HouseAge": 22.0,
                "AveRooms": 5.10,
                "AveBedrms": 1.05,
                "Population": 1600.0,
                "AveOccup": 3.10,
                "Latitude": 36.75,
                "Longitude": -119.77,
            },
        },
    ]
    sample_predictions = []
    for item in sample_profiles:
        pred_usd = predict_house_value(best_model, item["features"])
        sample_predictions.append(
            {
                "profile": item["profile"],
                "features": item["features"],
                "predicted_value_usd": pred_usd,
                "predicted_value_100k": round(pred_usd / 100_000.0, 4),
            }
        )

    generate_figures(
        df=df,
        y_test=y_test,
        metrics_list=metrics_list,
        best_model_name=best_model_name,
        best_pred=best_pred,
        perm_importance_df=perm_df,
        dataset_summary=dataset_summary,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "dataset_summary": dataset_summary,
        "models": metrics_list,
        "best_model": best_model_name,
        "permutation_importance": [
            {
                "feature": row["feature"],
                "importance_mean": round(float(row["importance_mean"]), 4),
                "importance_std": round(float(row["importance_std"]), 4),
            }
            for _, row in perm_df.iterrows()
        ],
        "sample_predictions": sample_predictions,
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("=== Dataset Summary ===")
    print(
        f"Samples: {dataset_summary['n_samples']} | Features: {dataset_summary['n_features']} | "
        f"Missing: {dataset_summary['missing_values_total']} | "
        f"Capped >= 5.0: {dataset_summary['capped_at_500k_count']} ({dataset_summary['capped_at_500k_pct']}%)"
    )
    print("\n=== Model Evaluation (80/20 Split, random_state=42) ===")
    for row in metrics_list:
        print(
            f"{row['model']:22s} | Test RMSE={row['test_rmse']:.4f} (${row['test_rmse_usd']:,.0f}) | "
            f"Test MAE={row['test_mae']:.4f} | Test R2={row['test_r2']:.4f} | "
            f"5-Fold CV R2={row['cv_r2_mean']:.4f} +/- {row['cv_r2_std']:.4f}"
        )
    print(f"\nBest model: {best_model_name} (saved to {model_path})")
    print("\n=== Single-Sample Inference Examples ===")
    for pred in sample_predictions:
        print(f"{pred['profile']}: ${pred['predicted_value_usd']:,.2f}")


if __name__ == "__main__":
    main()

