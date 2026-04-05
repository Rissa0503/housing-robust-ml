
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from model import get_models
from stress_utils import evaluate_regression


def _resolve_group_columns(columns: List[str], feature_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    resolved = {}
    for group_name, keywords in feature_groups.items():
        cols = [c for c in columns if any(k.lower() in c.lower() for k in keywords)]
        if cols:
            resolved[group_name] = cols
    return resolved


class PatternAdaptiveGradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """
    A lightweight finitely adaptive GB regressor inspired by Bertsimas et al.

    Idea:
    - fit one strong full-feature GB for the clean/full-information scenario
    - fit additional fallback GB models for specific missing-feature patterns
    - at prediction time, route the input to the matching fallback model based on
      which feature groups are unavailable

    This is not the exact adaptive linear-regression framework from the paper.
    It is a practical tabular analogue for a tree-based predictor.
    """

    def __init__(
        self,
        feature_groups: Dict[str, List[str]],
        use_pairwise_combos: bool = True,
        fill_strategy: str = "median",
        random_state: int = 42,
    ):
        self.feature_groups = feature_groups
        self.use_pairwise_combos = use_pairwise_combos
        self.fill_strategy = fill_strategy
        self.random_state = random_state

    def _get_base_model(self):
        return get_models(random_state=self.random_state)["GradientBoosting"]

    def _compute_fill_values(self, X: pd.DataFrame) -> pd.Series:
        if self.fill_strategy == "median":
            return X.median()
        elif self.fill_strategy == "mean":
            return X.mean()
        else:
            raise ValueError("fill_strategy must be either 'median' or 'mean'")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        y = y.copy()

        self.full_columns_ = list(X.columns)
        self.fill_values_ = self._compute_fill_values(X)
        self.group_columns_ = _resolve_group_columns(self.full_columns_, self.feature_groups)

        # Full model
        self.full_model_ = self._get_base_model()
        self.full_model_.fit(X[self.full_columns_], y)

        # Fallback models keyed by missing-group pattern
        self.fallback_models_ = {}
        self.fallback_feature_sets_ = {}

        # Single-group fallbacks
        for g, cols_to_remove in self.group_columns_.items():
            keep_cols = [c for c in self.full_columns_ if c not in cols_to_remove]
            m = self._get_base_model()
            m.fit(X[keep_cols], y)
            key = tuple(sorted([g]))
            self.fallback_models_[key] = m
            self.fallback_feature_sets_[key] = keep_cols

        # Pairwise fallbacks
        if self.use_pairwise_combos and len(self.group_columns_) >= 2:
            group_names = sorted(self.group_columns_.keys())
            for combo in combinations(group_names, 2):
                remove_cols = []
                for g in combo:
                    remove_cols.extend(self.group_columns_[g])
                remove_cols = sorted(set(remove_cols))
                keep_cols = [c for c in self.full_columns_ if c not in remove_cols]
                m = self._get_base_model()
                m.fit(X[keep_cols], y)
                key = tuple(sorted(combo))
                self.fallback_models_[key] = m
                self.fallback_feature_sets_[key] = keep_cols

        return self

    def _detect_missing_groups_from_columns(self, X: pd.DataFrame) -> Tuple[str, ...]:
        present = set(X.columns)
        missing_groups = []
        for g, cols in self.group_columns_.items():
            if any(c not in present for c in cols):
                missing_groups.append(g)
        return tuple(sorted(missing_groups))

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        missing_groups = self._detect_missing_groups_from_columns(X)

        if len(missing_groups) == 0:
            X_use = X.reindex(columns=self.full_columns_).fillna(self.fill_values_)
            return self.full_model_.predict(X_use)

        if missing_groups in self.fallback_models_:
            feat_cols = self.fallback_feature_sets_[missing_groups]
            X_use = X.reindex(columns=feat_cols)
            return self.fallback_models_[missing_groups].predict(X_use)

        # Fallback: if we do not have a dedicated model for this missingness pattern,
        # use the full model with simple imputation.
        X_use = X.reindex(columns=self.full_columns_).fillna(self.fill_values_)
        return self.full_model_.predict(X_use)


def fit_static_gb_baseline(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    model = get_models(random_state=random_state)["GradientBoosting"]
    model.fit(X_train, y_train)
    return model


def predict_static_with_group_imputation(
    model,
    X_scenario: pd.DataFrame,
    full_columns: List[str],
    fill_values: pd.Series,
):
    X_use = X_scenario.reindex(columns=full_columns).copy()
    X_use = X_use.fillna(fill_values)
    return model.predict(X_use)


def make_missing_scenario(X: pd.DataFrame, feature_groups: Dict[str, List[str]], groups_to_drop: List[str]) -> pd.DataFrame:
    X_scenario = X.copy()
    remove_cols = []
    for g in groups_to_drop:
        keywords = feature_groups[g]
        remove_cols.extend([c for c in X_scenario.columns if any(k.lower() in c.lower() for k in keywords)])
    remove_cols = sorted(set(remove_cols))
    return X_scenario.drop(columns=remove_cols, errors="ignore")


def evaluate_static_vs_adaptive(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    feature_groups: Dict[str, List[str]],
    use_pairwise_combos: bool = True,
    random_state: int = 42,
):
    static_model = fit_static_gb_baseline(X_train, y_train, random_state=random_state)
    adaptive_model = PatternAdaptiveGradientBoostingRegressor(
        feature_groups=feature_groups,
        use_pairwise_combos=use_pairwise_combos,
        random_state=random_state,
    ).fit(X_train, y_train)

    fill_values = X_train.median()
    full_columns = list(X_train.columns)

    scenarios = {
        "clean": [],
        "missing_garage": [g for g in feature_groups if g.lower() == "garage"],
        "missing_basement": [g for g in feature_groups if g.lower() == "basement"],
    }

    # pairwise missingness only if both exist
    if {"Garage", "Basement"}.issubset(set(feature_groups.keys())):
        scenarios["missing_garage_and_basement"] = ["Garage", "Basement"]

    rows = []

    for scenario_name, groups_to_drop in scenarios.items():
        if groups_to_drop:
            X_scenario = make_missing_scenario(X_valid, feature_groups, groups_to_drop)
        else:
            X_scenario = X_valid.copy()

        static_pred = predict_static_with_group_imputation(
            static_model, X_scenario, full_columns, fill_values
        )
        adaptive_pred = adaptive_model.predict(X_scenario)

        static_metrics = evaluate_regression(y_valid, static_pred)
        adaptive_metrics = evaluate_regression(y_valid, adaptive_pred)

        rows.append({
            "scenario": scenario_name,
            "model_type": "GB_StaticImpute",
            **static_metrics,
        })
        rows.append({
            "scenario": scenario_name,
            "model_type": "GB_FinitelyAdaptive",
            **adaptive_metrics,
        })

    result_df = pd.DataFrame(rows)
    pivot_df = result_df.pivot(index="scenario", columns="model_type", values="rmse").reset_index()
    if {"GB_StaticImpute", "GB_FinitelyAdaptive"}.issubset(set(result_df["model_type"].unique())):
        pivot_df["adaptive_minus_static_rmse"] = pivot_df["GB_FinitelyAdaptive"] - pivot_df["GB_StaticImpute"]

    return result_df, pivot_df, static_model, adaptive_model
