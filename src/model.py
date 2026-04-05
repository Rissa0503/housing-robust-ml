# src/model.py

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def rmse_cv(model, X, y, n_folds=5, random_state=42):
    """
    Cross-validated RMSE on the current target.
    If y is log1p(SalePrice), this is RMSE in log-space.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    rmse = np.sqrt(
        -cross_val_score(
            model,
            X,
            y,
            scoring="neg_mean_squared_error",
            cv=kf,
            n_jobs=None
        )
    )
    return rmse


def get_models(random_state=42) -> Dict[str, Any]:
    """
    Return a dictionary of candidate models.
    Models that fail to import are skipped automatically.
    """
    models: Dict[str, Any] = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=15.0))
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.005, random_state=random_state, max_iter=100000, tol=1e-3))
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.005, l1_ratio=0.9, random_state=random_state, max_iter=100000, tol=1e-3))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=8,
            subsample=0.7,
            max_features="sqrt",
            random_state=random_state
        ),
    }

    # LightGBM (optional)
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=3,
            num_leaves=8,
            min_child_samples=30,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
    except Exception as e:
        print(f"[Info] LightGBM not available: {repr(e)}")

    # XGBoost (optional)
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=3,
            gamma=0.1,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=0.0001,
            reg_lambda=2.0,
            random_state=random_state,
            n_jobs=-1,
            objective="reg:squarederror"
        )
    except Exception as e:
        print(f"[Info] XGBoost not available: {repr(e)}")

    return models


def compare_models_cv(X, y, n_folds=5, random_state=42) -> pd.DataFrame:
    """
    Run cross-validation for all available models and return a detailed table:
    - rank
    - RMSE mean / std / min / max
    - RMSE for each fold
    """
    models = get_models(random_state=random_state)
    rows = []

    for name, model in models.items():
        scores = rmse_cv(model, X, y, n_folds=n_folds, random_state=random_state)

        row = {
            "model": name,
            "rmse_mean": float(scores.mean()),
            "rmse_std": float(scores.std()),
            "rmse_min": float(scores.min()),
            "rmse_max": float(scores.max()),
        }

        for i, score in enumerate(scores, start=1):
            row[f"fold_{i}_rmse"] = float(score)

        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)
    result_df.insert(0, "rank", range(1, len(result_df) + 1))
    return result_df


def get_best_model_name(results_df: pd.DataFrame) -> str:
    """
    Return the best model name based on the smallest mean CV RMSE.
    """
    return results_df.iloc[0]["model"]


def print_model_results(results_df: pd.DataFrame, decimals: int = 5) -> None:
    """
    Print the full comparison table nicely in notebook/terminal.
    """
    print(results_df.round(decimals).to_string(index=False))


def plot_model_comparison(results_df: pd.DataFrame, figsize=(10, 5)):
    """
    Plot mean CV RMSE with error bars (std).
    Lower is better.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install it with: %pip install matplotlib"
        ) from e

    plot_df = results_df.copy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        plot_df["model"],
        plot_df["rmse_mean"],
        yerr=plot_df["rmse_std"],
        capsize=4
    )
    ax.set_title("Model Comparison (CV RMSE)")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig, ax


def export_model_results(results_df: pd.DataFrame, filepath: str = "outputs/model_cv_results.csv") -> Path:
    """
    Export model comparison table to CSV.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)
    print(f"[Saved] {path.resolve()}")
    return path


def fit_best_model(model_name: str, X, y, random_state=42):
    """
    Fit a named model from the candidate pool.
    """
    models = get_models(random_state=random_state)
    model = models[model_name]
    model.fit(X, y)
    return model


def fit_best_from_cv(X, y, n_folds=5, random_state=42) -> Tuple[Any, str, pd.DataFrame]:
    """
    1) Compare all models with CV
    2) Select the best one
    3) Fit it on the full training data
    Returns:
        best_model, best_model_name, results_df
    """
    results_df = compare_models_cv(X, y, n_folds=n_folds, random_state=random_state)
    best_model_name = get_best_model_name(results_df)
    best_model = fit_best_model(best_model_name, X, y, random_state=random_state)
    return best_model, best_model_name, results_df

def seed_sensitivity_analysis(X, y, seeds=(0, 21, 42, 84, 126), n_folds=5):
    """
    Re-run model comparison under multiple random seeds.
    Returns:
        all_results: long-format results for every seed and model
        summary: aggregated summary across seeds
    """
    frames = []

    for seed in seeds:
        df = compare_models_cv(X, y, n_folds=n_folds, random_state=seed).copy()
        df["seed"] = seed
        frames.append(df)

    all_results = pd.concat(frames, ignore_index=True)

    summary = (
        all_results.groupby("model")
        .agg(
            mean_rmse=("rmse_mean", "mean"),
            std_across_seeds=("rmse_mean", "std"),
            best_rmse=("rmse_mean", "min"),
            worst_rmse=("rmse_mean", "max"),
        )
        .sort_values("mean_rmse")
        .reset_index()
    )

    winners = (
        all_results.loc[all_results.groupby("seed")["rmse_mean"].idxmin(), "model"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="win_count")
    )

    summary = summary.merge(winners, on="model", how="left").fillna({"win_count": 0})
    summary["win_count"] = summary["win_count"].astype(int)

    return all_results, summary

def fold_sensitivity_analysis(X, y, fold_list=(5, 10), random_state=42):
    """
    Compare model rankings under different CV fold settings.
    Returns:
        fold_results: long-format results
        fold_compare: wide-format comparison table
    """
    fold_frames = []

    for n_folds in fold_list:
        df = compare_models_cv(X, y, n_folds=n_folds, random_state=random_state).copy()
        df["n_folds"] = n_folds
        fold_frames.append(df)

    fold_results = pd.concat(fold_frames, ignore_index=True)

    fold_compare = (
        fold_results.pivot(index="model", columns="n_folds", values="rmse_mean")
        .rename(columns={k: f"rmse_{k}fold" for k in fold_list})
    )

    if 5 in fold_list and 10 in fold_list:
        fold_compare["delta_10_minus_5"] = (
            fold_compare["rmse_10fold"] - fold_compare["rmse_5fold"]
        )

    first_col = fold_compare.columns[0]
    fold_compare = fold_compare.sort_values(first_col).reset_index()

    return fold_results, fold_compare

def add_small_noise_to_numeric(X, noise_ratio=0.01, random_state=42):
    """
    Add small Gaussian noise to numeric columns only.
    noise_ratio is relative to each column's standard deviation.
    """
    X_noisy = X.copy()
    rng = np.random.default_rng(random_state)

    num_cols = X_noisy.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        std = X_noisy[col].std()
        if pd.notnull(std) and std > 0:
            noise = rng.normal(loc=0, scale=noise_ratio * std, size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise

    return X_noisy

def noise_robustness_analysis(X, y, noise_ratio=0.01, n_folds=5, random_state=42):
    """
    Compare model performance before and after adding small feature noise.
    Returns:
        clean_results: original CV results
        noisy_results: noisy-feature CV results
        noise_compare: merged comparison table
    """
    X_noisy = add_small_noise_to_numeric(
        X, noise_ratio=noise_ratio, random_state=random_state
    )

    clean_results = compare_models_cv(X, y, n_folds=n_folds, random_state=random_state)
    noisy_results = compare_models_cv(X_noisy, y, n_folds=n_folds, random_state=random_state)

    noise_compare = clean_results[["model", "rmse_mean"]].merge(
        noisy_results[["model", "rmse_mean"]],
        on="model",
        suffixes=("_clean", "_noisy")
    )

    noise_compare["delta_rmse"] = (
        noise_compare["rmse_mean_noisy"] - noise_compare["rmse_mean_clean"]
    )

    noise_compare = noise_compare.sort_values("delta_rmse").reset_index(drop=True)

    return clean_results, noisy_results, noise_compare