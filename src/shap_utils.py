# src/shap.py

from typing import Tuple, List
import pandas as pd
import shap
import matplotlib.pyplot as plt


def build_shap_explainer(model, X_background):
    """
    Build a SHAP explainer for a fitted tree-based or linear-compatible model.
    """
    try:
        explainer = shap.Explainer(model, X_background)
    except Exception:
        # fallback
        explainer = shap.TreeExplainer(model)
    return explainer


def compute_shap_values(explainer, X_sample):
    """
    Compute SHAP values for a given sample.
    """
    shap_values = explainer(X_sample)
    return shap_values


def plot_shap_bar(shap_values, max_display=15):
    """
    Global feature importance bar plot.
    """
    shap.plots.bar(shap_values, max_display=max_display)


def plot_shap_beeswarm(shap_values, max_display=15):
    """
    Global beeswarm summary plot.
    """
    shap.plots.beeswarm(shap_values, max_display=max_display)


def get_mean_abs_shap(shap_values, feature_names=None) -> pd.DataFrame:
    """
    Return a dataframe of mean absolute SHAP importance.
    """
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 0]

    mean_abs = abs(values).mean(axis=0)

    if feature_names is None:
        feature_names = shap_values.feature_names

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return df


def plot_shap_dependence(shap_values, feature: str, X_sample: pd.DataFrame):
    """
    Dependence plot for one feature.
    """
    shap.plots.scatter(shap_values[:, feature], color=shap_values)


def get_top_features(shap_importance_df: pd.DataFrame, top_n=10) -> List[str]:
    return shap_importance_df.head(top_n)["feature"].tolist()