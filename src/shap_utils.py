# shap_utils.py
from __future__ import annotations

from typing import List
import pandas as pd
import shap


def _unwrap_model_for_shap(model):
    """
    Return the underlying fitted estimator when the saved model is a lightweight
    wrapper class (e.g. RobustGradientBoostingRegressor with attribute `model_`).
    """
    if hasattr(model, "model_"):
        return model.model_
    return model


def build_shap_explainer(model, X_background: pd.DataFrame):
    """
    Build a SHAP explainer for the final selected model.

    This function first unwraps custom wrapper models used in the intervention
    notebook. If a direct tree/linear explainer is unavailable, it falls back to
    a model-agnostic explainer based on the model's predict function.
    """
    base_model = _unwrap_model_for_shap(model)

    try:
        return shap.Explainer(base_model, X_background)
    except Exception:
        pass

    try:
        return shap.TreeExplainer(base_model)
    except Exception:
        pass

    feature_names = list(X_background.columns)

    def predict_fn(x):
        x_df = pd.DataFrame(x, columns=feature_names)
        return model.predict(x_df)

    return shap.Explainer(predict_fn, X_background)


def compute_shap_values(explainer, X_sample: pd.DataFrame):
    """
    Compute SHAP values for a given sample.
    """
    return explainer(X_sample)


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

    return (
        pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def get_top_features(shap_importance_df: pd.DataFrame, top_n=10) -> List[str]:
    return shap_importance_df.head(top_n)["feature"].tolist()
