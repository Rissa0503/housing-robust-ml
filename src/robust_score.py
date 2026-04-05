import numpy as np
import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    series = pd.Series(series)
    std = series.std(ddof=0)
    if std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def build_robust_summary(
    clean_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    family_weights=None,
    clean_weight: float = 1.0,
    robust_weight: float = 0.5,
):
    """
    Build a robustness-aware summary table.

    clean_weight controls how much clean hold-out performance matters.
    robust_weight controls how much the robustness penalty matters.
    """

    if family_weights is None:
        family_weights = {
            "feature_ablation": 0.40,
            "feature_noise": 0.20,
            "target_corruption": 0.15,
            "worst_group_gap": 0.25,
        }

    clean_summary = clean_df.copy()

    family_wide = (
        family_summary_df.pivot(index="model", columns="stress_family", values="mean_delta_rmse")
        .reset_index()
    )
    family_wide.columns.name = None

    subgroup_summary = (
        subgroup_df.groupby("model", as_index=False)
        .agg(
            worst_group_rmse=("rmse", "max"),
            best_group_rmse=("rmse", "min"),
            mean_group_rmse=("rmse", "mean"),
            worst_group_gap=("rmse_gap_vs_overall", "max"),
        )
    )

    summary = clean_summary.merge(family_wide, on="model", how="left").merge(subgroup_summary, on="model", how="left")

    for col in ["feature_ablation", "feature_noise", "target_corruption", "worst_group_gap"]:
        if col not in summary.columns:
            summary[col] = 0.0

    summary["robust_penalty_raw"] = (
        family_weights.get("feature_ablation", 0.0) * summary["feature_ablation"].fillna(0.0)
        + family_weights.get("feature_noise", 0.0) * summary["feature_noise"].fillna(0.0)
        + family_weights.get("target_corruption", 0.0) * summary["target_corruption"].fillna(0.0)
        + family_weights.get("worst_group_gap", 0.0) * summary["worst_group_gap"].fillna(0.0)
    )

    summary["RobustScore"] = -summary["robust_penalty_raw"]

    summary["joint_rank_score"] = (
        clean_weight * zscore(summary["clean_rmse"])
        + robust_weight * zscore(summary["robust_penalty_raw"])
    )

    summary = summary.sort_values("joint_rank_score", ascending=True).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary
