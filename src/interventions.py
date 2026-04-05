
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from model import get_models
from stress_utils import add_numeric_noise


class TargetedRobustGradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """
    Lightweight GradientBoosting wrapper for targeted robustness interventions.

    This class does not change the boosting algorithm itself. It changes the
    empirical training problem through three optional mechanisms:

    1) noise augmentation on a subset of rows
    2) feature-group masking augmentation with explicit missing-group indicators
    3) coupled hybrid augmentation that applies masking and noise together
    """

    def __init__(
        self,
        noise_cols: Optional[Sequence[str]] = None,
        feature_groups: Optional[Dict[str, Sequence[str]]] = None,
        noise_scale: float = 0.03,
        n_noise_copies: int = 0,
        noise_row_frac: float = 1.0,
        noise_copy_weight: float = 0.60,
        n_mask_copies: int = 0,
        mask_row_frac: float = 0.35,
        mask_copy_weight: float = 0.45,
        mask_fill: str = "median",
        mask_groups_per_copy: int = 1,
        random_masking: bool = True,
        add_missing_indicators: bool = True,
        n_hybrid_copies: int = 0,
        hybrid_row_frac: float = 0.45,
        hybrid_copy_weight: float = 0.55,
        group_weighting: bool = False,
        q: int = 3,
        random_state: int = 42,
    ):
        self.noise_cols = noise_cols
        self.feature_groups = feature_groups
        self.noise_scale = noise_scale
        self.n_noise_copies = n_noise_copies
        self.noise_row_frac = noise_row_frac
        self.noise_copy_weight = noise_copy_weight
        self.n_mask_copies = n_mask_copies
        self.mask_row_frac = mask_row_frac
        self.mask_copy_weight = mask_copy_weight
        self.mask_fill = mask_fill
        self.mask_groups_per_copy = mask_groups_per_copy
        self.random_masking = random_masking
        self.add_missing_indicators = add_missing_indicators
        self.n_hybrid_copies = n_hybrid_copies
        self.hybrid_row_frac = hybrid_row_frac
        self.hybrid_copy_weight = hybrid_copy_weight
        self.group_weighting = group_weighting
        self.q = q
        self.random_state = random_state

    def _get_base_model(self):
        return get_models(random_state=self.random_state)["GradientBoosting"]

    @staticmethod
    def _indicator_name(group_name: str) -> str:
        return f"MISSGRP__{group_name}"

    def _resolve_group_columns(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        group_to_cols: Dict[str, List[str]] = {}
        feature_groups = self.feature_groups or {}
        for group_name, keywords in feature_groups.items():
            cols = [
                c for c in X.columns
                if any(k.lower() in c.lower() for k in keywords)
            ]
            cols = [c for c in cols if c in X.columns]
            if cols:
                group_to_cols[group_name] = cols
        return group_to_cols

    def _compute_fill_values(self, X: pd.DataFrame) -> Dict[str, float]:
        fill_values: Dict[str, float] = {}
        for c in X.columns:
            if self.mask_fill == "median":
                val = X[c].median()
                if pd.isna(val):
                    val = 0.0
                fill_values[c] = float(val)
            elif self.mask_fill == "zero":
                fill_values[c] = 0.0
            else:
                raise ValueError("mask_fill must be 'median' or 'zero'.")
        return fill_values

    def _add_indicator_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.add_missing_indicators:
            for group_name in self.group_to_cols_.keys():
                ind = self._indicator_name(group_name)
                if ind not in X_out.columns:
                    X_out[ind] = 0.0
        return X_out

    def _sample_rows(self, X: pd.DataFrame, y: pd.Series, frac: float, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.Series]:
        frac = float(frac)
        if frac >= 1.0:
            return X.copy(), y.copy()
        n = len(X)
        k = max(1, int(round(frac * n)))
        idx = rng.choice(np.arange(n), size=k, replace=False)
        idx = np.sort(idx)
        return X.iloc[idx].copy(), y.iloc[idx].copy()

    def _choose_groups_for_copy(self, copy_idx: int, rng: np.random.Generator) -> List[str]:
        group_names = [g for g, cols in self.group_to_cols_.items() if len(cols) > 0]
        if not group_names:
            return []
        k = max(1, min(self.mask_groups_per_copy, len(group_names)))
        if self.random_masking:
            return list(rng.choice(group_names, size=k, replace=False))
        start = (copy_idx * k) % len(group_names)
        return [group_names[(start + j) % len(group_names)] for j in range(k)]

    def _apply_group_mask(self, X: pd.DataFrame, selected_groups: Sequence[str]) -> pd.DataFrame:
        X_masked = X.copy()
        for group_name in selected_groups:
            cols = self.group_to_cols_.get(group_name, [])
            for c in cols:
                if c in X_masked.columns:
                    X_masked[c] = self.fill_values_[c]
            if self.add_missing_indicators:
                ind = self._indicator_name(group_name)
                if ind in X_masked.columns:
                    X_masked[ind] = 1.0
        return X_masked

    def _build_sample_weight(self, y: pd.Series):
        # label-derived subgroup weighting, kept optional
        bins = pd.qcut(np.expm1(y), q=self.q, labels=False, duplicates='drop')
        groups = pd.Series(bins, index=y.index)
        counts = groups.value_counts()
        weights = groups.map(lambda g: 1.0 / counts[g]).astype(float)
        weights = weights / weights.mean()
        return weights.to_numpy()

    def _prepare_predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X_pred = X.copy()
        X_pred = self._add_indicator_columns(X_pred)
        for c in self.train_columns_:
            if c not in X_pred.columns:
                X_pred[c] = 0.0
        X_pred = X_pred[self.train_columns_]
        return X_pred

    def _build_augmented_training_set(self, X: pd.DataFrame, y: pd.Series):
        rng = np.random.default_rng(self.random_state)
        X_parts = [X.copy()]
        y_parts = [y.copy()]
        w_parts = [np.ones(len(X), dtype=float)]

        cols = list(self.noise_cols) if self.noise_cols is not None else []

        # A) gentler local perturbation copies on a subset of rows
        for i in range(self.n_noise_copies):
            X_sub, y_sub = self._sample_rows(X, y, self.noise_row_frac, rng)
            X_noisy = add_numeric_noise(
                X_sub,
                cols=cols,
                noise_scale=self.noise_scale,
                random_state=self.random_state + 31 * (i + 1),
            )
            X_parts.append(X_noisy)
            y_parts.append(y_sub)
            w_parts.append(np.full(len(X_noisy), self.noise_copy_weight, dtype=float))

        # B) explicit feature-group masking copies on a subset of rows
        for i in range(self.n_mask_copies):
            X_sub, y_sub = self._sample_rows(X, y, self.mask_row_frac, rng)
            selected_groups = self._choose_groups_for_copy(i, rng)
            if not selected_groups:
                continue
            X_masked = self._apply_group_mask(X_sub, selected_groups)
            X_parts.append(X_masked)
            y_parts.append(y_sub)
            w_parts.append(np.full(len(X_masked), self.mask_copy_weight, dtype=float))

        # C) coupled hybrid copies: noise + masking on the same rows
        for i in range(self.n_hybrid_copies):
            X_sub, y_sub = self._sample_rows(X, y, self.hybrid_row_frac, rng)
            selected_groups = self._choose_groups_for_copy(i, rng)
            if selected_groups:
                X_h = self._apply_group_mask(X_sub, selected_groups)
                masked_cols = set(sum([self.group_to_cols_.get(g, []) for g in selected_groups], []))
            else:
                X_h = X_sub.copy()
                masked_cols = set()
            effective_noise_cols = [c for c in cols if c in X_h.columns and c not in masked_cols]
            X_h = add_numeric_noise(
                X_h,
                cols=effective_noise_cols,
                noise_scale=self.noise_scale,
                random_state=self.random_state + 101 * (i + 1),
            )
            X_parts.append(X_h)
            y_parts.append(y_sub)
            w_parts.append(np.full(len(X_h), self.hybrid_copy_weight, dtype=float))

        X_aug = pd.concat(X_parts, axis=0).reset_index(drop=True)
        y_aug = pd.concat(y_parts, axis=0).reset_index(drop=True)
        w_aug = np.concatenate(w_parts)
        return X_aug, y_aug, w_aug

    def fit(self, X, y):
        X_fit = X.copy()
        y_fit = y.copy()

        self.raw_feature_names_ = list(X_fit.columns)
        self.group_to_cols_ = self._resolve_group_columns(X_fit)
        self.fill_values_ = self._compute_fill_values(X_fit)

        X_fit = self._add_indicator_columns(X_fit)
        X_fit, y_fit, aug_weight = self._build_augmented_training_set(X_fit, y_fit)

        if self.group_weighting:
            subgroup_weight = self._build_sample_weight(y_fit)
            sample_weight = aug_weight * subgroup_weight
        else:
            sample_weight = aug_weight

        model = self._get_base_model()
        model.fit(X_fit, y_fit, sample_weight=sample_weight)

        self.model_ = model
        self.train_columns_ = list(X_fit.columns)
        return self

    def predict(self, X):
        X_pred = self._prepare_predict_frame(X)
        return self.model_.predict(X_pred)


def get_gb_feature_missing_variants(noise_cols, feature_groups, random_state: int = 42):
    """
    One focused comparison with four pre-designed GB variants.
    The interventions are intentionally gentler than the earlier version,
    so they have a fair chance to improve robustness without paying too much
    clean-accuracy cost.
    """
    return {
        "GB_Baseline": TargetedRobustGradientBoostingRegressor(
            noise_cols=noise_cols,
            feature_groups=feature_groups,
            n_noise_copies=0,
            n_mask_copies=0,
            n_hybrid_copies=0,
            random_state=random_state,
        ),
        "GB_NoiseAug": TargetedRobustGradientBoostingRegressor(
            noise_cols=noise_cols,
            feature_groups=feature_groups,
            noise_scale=0.025,
            n_noise_copies=1,
            noise_row_frac=0.65,
            noise_copy_weight=0.60,
            n_mask_copies=0,
            n_hybrid_copies=0,
            random_state=random_state,
        ),
        "GB_FeatureMask": TargetedRobustGradientBoostingRegressor(
            noise_cols=noise_cols,
            feature_groups=feature_groups,
            n_noise_copies=0,
            n_mask_copies=2,
            mask_row_frac=0.30,
            mask_copy_weight=0.40,
            mask_groups_per_copy=1,
            random_masking=True,
            add_missing_indicators=True,
            n_hybrid_copies=0,
            random_state=random_state,
        ),
        "GB_NoisePlusFeatureMask": TargetedRobustGradientBoostingRegressor(
            noise_cols=noise_cols,
            feature_groups=feature_groups,
            noise_scale=0.025,
            n_noise_copies=0,
            n_mask_copies=0,
            n_hybrid_copies=2,
            hybrid_row_frac=0.45,
            hybrid_copy_weight=0.55,
            mask_groups_per_copy=1,
            random_masking=True,
            add_missing_indicators=True,
            random_state=random_state,
        ),
    }
