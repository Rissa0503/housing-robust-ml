# stress_protocol.py
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.stress_utils import (
    fit_and_eval,
    compare_metric_dicts,
    add_numeric_noise,
    drop_feature_group,
    make_price_groups,
    subgroup_evaluation,
    corrupt_targets,
    collect_stress_results,
)

def run_clean_benchmark(models, X_train, y_train, X_valid, y_valid):
    rows = []
    fitted_models = {}

    for model_name, model in models.items():
        fitted_model, pred, metrics = fit_and_eval(
            model, X_train, y_train, X_valid, y_valid
        )
        fitted_models[model_name] = fitted_model

        rows.append({
            "model": model_name,
            "test": "clean",
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        })

    return pd.DataFrame(rows), fitted_models


def run_model_stress_suite(
    model_name,
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    noise_cols,
    feature_groups,
    noise_scales=(0.01, 0.03, 0.05),
    target_scales=(0.03, 0.05),
    target_frac=0.10,
    random_state=42,
):
    result_list = []

    # 1) clean baseline
    fitted_model, _, base_metrics = fit_and_eval(model, X_train, y_train, X_valid, y_valid)

    # 2) test-time numeric noise
    for scale in noise_scales:
        X_valid_noisy = add_numeric_noise(
            X_valid, cols=noise_cols, noise_scale=scale, random_state=random_state
        )
        pred = fitted_model.predict(X_valid_noisy)
        stressed_metrics = {
            "rmse": np.sqrt(np.mean((y_valid - pred) ** 2)),
            "mae": np.mean(np.abs(y_valid - pred)),
            "r2": 1 - np.sum((y_valid - pred) ** 2) / np.sum((y_valid - np.mean(y_valid)) ** 2)
        }
        df = compare_metric_dicts(base_metrics, stressed_metrics, f"noise_{scale:.2f}")
        df["model"] = model_name
        df["stress_family"] = "feature_noise"
        result_list.append(df)

    # 3) feature-group ablation
    for group_name, keywords in feature_groups.items():
        X_train_drop = drop_feature_group(X_train, keywords)
        X_valid_drop = drop_feature_group(X_valid, keywords)

        _, _, stressed_metrics = fit_and_eval(
            model, X_train_drop, y_train, X_valid_drop, y_valid
        )
        df = compare_metric_dicts(base_metrics, stressed_metrics, f"drop_{group_name}")
        df["model"] = model_name
        df["stress_family"] = "feature_ablation"
        result_list.append(df)

    # 4) noisy target stress (train-time label-side stress for regression)
    for scale in target_scales:
        y_train_corrupt = corrupt_targets(
            y_train, mode="gaussian", scale=scale, random_state=random_state
        )
        _, _, stressed_metrics = fit_and_eval(
            model, X_train, y_train_corrupt, X_valid, y_valid
        )
        df = compare_metric_dicts(base_metrics, stressed_metrics, f"target_gaussian_{scale:.2f}")
        df["model"] = model_name
        df["stress_family"] = "target_corruption"
        result_list.append(df)

    y_train_subset_corrupt = corrupt_targets(
        y_train, mode="subset", scale=0.05, frac=target_frac, random_state=random_state
    )
    _, _, stressed_metrics = fit_and_eval(
        model, X_train, y_train_subset_corrupt, X_valid, y_valid
    )
    df = compare_metric_dicts(base_metrics, stressed_metrics, f"target_subset_{target_frac:.2f}")
    df["model"] = model_name
    df["stress_family"] = "target_corruption"
    result_list.append(df)

    # 5) subgroup evaluation on clean fitted model
    groups = make_price_groups(y_valid, q=3)
    subgroup_df = subgroup_evaluation(
        fitted_model, X_valid, y_valid, groups=groups, group_name="price_group"
    )
    subgroup_df["model"] = model_name
    subgroup_df["base_rmse"] = base_metrics["rmse"]
    subgroup_df["rmse_gap_vs_overall"] = subgroup_df["rmse"] - base_metrics["rmse"]

    stress_df = collect_stress_results(result_list)
    return stress_df, subgroup_df


def run_all_models_stress_suite(
    models,
    X_train,
    y_train,
    X_valid,
    y_valid,
    noise_cols,
    feature_groups,
    noise_scales=(0.01, 0.03, 0.05),
    target_scales=(0.03, 0.05),
    target_frac=0.10,
    random_state=42,
):
    stress_frames = []
    subgroup_frames = []

    for model_name, model in models.items():
        stress_df, subgroup_df = run_model_stress_suite(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            noise_cols=noise_cols,
            feature_groups=feature_groups,
            noise_scales=noise_scales,
            target_scales=target_scales,
            target_frac=target_frac,
            random_state=random_state,
        )
        stress_frames.append(stress_df)
        subgroup_frames.append(subgroup_df)

    return (
        pd.concat(stress_frames, axis=0).reset_index(drop=True),
        pd.concat(subgroup_frames, axis=0).reset_index(drop=True),
    )