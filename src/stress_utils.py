# src/stress_utils.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def make_train_valid_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def fit_and_eval(model, X_train, y_train, X_valid, y_valid):
    m = clone(model)
    m.fit(X_train, y_train)
    pred = m.predict(X_valid)
    metrics = evaluate_regression(y_valid, pred)
    return m, pred, metrics


def compare_metric_dicts(base_metrics, stressed_metrics, test_name):
    return pd.DataFrame([{
        "test": test_name,
        "base_rmse": base_metrics["rmse"],
        "stressed_rmse": stressed_metrics["rmse"],
        "delta_rmse": stressed_metrics["rmse"] - base_metrics["rmse"],
        "base_mae": base_metrics["mae"],
        "stressed_mae": stressed_metrics["mae"],
        "delta_mae": stressed_metrics["mae"] - base_metrics["mae"],
        "base_r2": base_metrics["r2"],
        "stressed_r2": stressed_metrics["r2"],
        "delta_r2": stressed_metrics["r2"] - base_metrics["r2"],
    }])


def add_numeric_noise(X, cols, noise_scale=0.03, random_state=42):
    rng = np.random.default_rng(random_state)
    X_noisy = X.copy()

    for col in cols:
        if col in X_noisy.columns:
            std = X_noisy[col].std()
            noise = rng.normal(loc=0.0, scale=noise_scale * std, size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise

    return X_noisy


def drop_feature_group(X, keywords):
    keep_cols = []
    for c in X.columns:
        if not any(k.lower() in c.lower() for k in keywords):
            keep_cols.append(c)
    return X[keep_cols].copy()


def make_price_groups(y_valid_log, q=3):
    # convert back to original scale for more interpretable grouping
    price = np.expm1(y_valid_log)
    return pd.qcut(price, q=q, labels=["low", "mid", "high"])


def subgroup_evaluation(model, X_valid, y_valid, groups, group_name="group"):
    pred = model.predict(X_valid)
    eval_df = pd.DataFrame({
        "y_true": y_valid,
        "y_pred": pred,
        group_name: groups
    })

    rows = []
    for g in eval_df[group_name].dropna().unique():
        sub = eval_df[eval_df[group_name] == g]
        metrics = evaluate_regression(sub["y_true"], sub["y_pred"])
        rows.append({
            group_name: g,
            "count": len(sub),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"]
        })

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def corrupt_targets(y, mode="gaussian", scale=0.03, frac=0.1, random_state=42):
    """
    Create a corrupted training target to simulate imperfect supervision.

    mode='gaussian': add Gaussian noise to all targets
    mode='subset': corrupt only a subset of targets
    """
    rng = np.random.default_rng(random_state)
    y_corrupt = y.copy().astype(float)

    if mode == "gaussian":
        std = y_corrupt.std()
        y_corrupt = y_corrupt + rng.normal(0.0, scale * std, size=len(y_corrupt))

    elif mode == "subset":
        n = len(y_corrupt)
        k = int(frac * n)
        idx = rng.choice(np.arange(n), size=k, replace=False)
        std = y_corrupt.std()
        y_corrupt.iloc[idx] = y_corrupt.iloc[idx] + rng.normal(0.0, scale * std, size=k)

    return y_corrupt


def collect_stress_results(result_list):
    return pd.concat(result_list, axis=0).reset_index(drop=True)