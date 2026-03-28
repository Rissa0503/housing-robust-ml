# src/feature.py

import numpy as np
import pandas as pd


def remove_outliers(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove a small number of highly unusual observations
    based on domain logic and visual inspection.
    """
    df = train_df.copy()

    # 1. Very large living area but abnormally low sale price
    df = df.drop(df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)].index)

    # 2. Extremely large basement area
    df = df.drop(df[(df["TotalBsmtSF"] > 3000)].index)

    # 3. Extremely large lot area
    df = df.drop(df[(df["LotArea"] > 100000)].index)

    return df


def impute_missing_values(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values based on semantic meaning and feature type.
    """
    df = all_data.copy()

    # Pattern 1: "None" means absence of the feature
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType", "MSSubClass"
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Pattern 2: 0 means absence for numeric variables
    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Pattern 3: mode imputation for selected categorical variables
    mode_cols = [
        "MSZoning", "Electrical", "KitchenQual", "Exterior1st",
        "Exterior2nd", "SaleType", "Functional"
    ]
    for col in mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Pattern 4: group-based median imputation
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

    return df


def add_engineered_features(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a selected set of domain-informed engineered features.
    """
    df = all_data.copy()

    # Aggregation
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"] +
        df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] +
        df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]
    )

    # Temporal
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["YearsSinceRemod"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]

    # Binary
    df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
    df["HasRemod"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    # Interaction
    df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]
    df["OverallQual_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]

    # Ratio
    df["Bsmt_Ratio"] = df["TotalBsmtSF"] / (df["TotalSF"] + 1)
    df["AreaPerRoom"] = df["GrLivArea"] / (df["TotRmsAbvGrd"] + 1)

    return df


def encode_features(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply one-hot encoding as a baseline categorical encoding strategy.
    """
    df = all_data.copy()
    df = pd.get_dummies(df, drop_first=True)
    return df