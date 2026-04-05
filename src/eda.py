import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set_theme()

def quick_eda(df: pd.DataFrame, target_col: str):
    print("--- 1. dataframe ---")
    print(f"shape: {df.shape}")
    print("\ntype（20）：")
    print(df.dtypes.head(20))

    print("\n--- 2. target_col ---")
    print(df[target_col].describe())
    plt.figure()
    sns.histplot(df[target_col], bins=30, kde=True)
    plt.title(f"{target_col} distribution")
    plt.show()

    if (df[target_col] > 0).all():
        plt.figure()
        sns.histplot(np.log1p(df[target_col]), bins=30, kde=True)
        plt.title(f"log1p({target_col}) distribution")
        plt.show()

    print("\n--- 3. missing_count（20） ---")
    missing_count = df.isna().sum()
    missing_ratio = missing_count / len(df)
    miss_df = pd.DataFrame({
    "missing_count": missing_count,
    "missing_ratio": missing_ratio
    }).sort_values("missing_ratio", ascending=False)
    print(miss_df[miss_df["missing_count"] > 0].head(20))

    print("\n--- 4. corr_to_target（20） ---")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        corr_to_target = df[num_cols].corr()[target_col].drop(target_col)
        corr_to_target = corr_to_target.reindex(
            corr_to_target.abs().sort_values(ascending=False).index
        )
        print(corr_to_target.head(20))
    else:
        print(f"{target_col} not a numeric type, so Pearson correlation cannot be performed.")

    print("\n--- 5. cat_cols&exp ---")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Number of categorical features: {len(cat_cols)}")
    if len(cat_cols) > 0:
        print("exp（The first few items of the value_counts for the first 5 category features）：")
        for c in cat_cols[:5]:
            print(f"\n[{c}]")
            print(df[c].value_counts().head())

def plot_top_corr_scatter(df, target_col, top_n=4):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    corr_to_target = df[num_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    top_cols = corr_to_target.head(top_n).index

    for c in top_cols:
        plt.figure()
        sns.scatterplot(data=df, x=c, y=target_col)
        plt.title(f"{c} vs {target_col}")
        plt.show()


def plot_cat_vs_target(df, cat_col, target_col, top_k=15):
    group = df.groupby(cat_col)[target_col].mean().sort_values(ascending=False)
    group = group.head(top_k)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=group.index, y=group.values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(cat_col)
    plt.ylabel(f"mean {target_col}")
    plt.title(f"{cat_col} vs mean {target_col}")
    plt.tight_layout()
    plt.show()

