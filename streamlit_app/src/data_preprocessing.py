import pandas as pd
import numpy as np

def preprocess(df, is_train=True, feature_columns=None):
    df = df.copy()

    # -------------------------
    # Keep Name TEMPORARILY (needed for Title)
    # -------------------------

    # -------------------------
    # Handle missing values
    # -------------------------
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # -------------------------
    # Feature engineering
    # -------------------------
    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # -------------------------
    # Title extraction (HIGH IMPACT)
    # -------------------------
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    else:
        df["Title"] = "Unknown"

    df["Title"] = df["Title"].replace(
        ["Lady","Countess","Capt","Col","Don","Dr",
         "Major","Rev","Sir","Jonkheer","Dona"],
        "Rare"
    )
    df["Title"] = df["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )

    # -------------------------
    # DROP NON-FEATURE COLUMNS (AFTER feature creation)
    # -------------------------
    df.drop(
        columns=["PassengerId", "Name", "Ticket", "Cabin"],
        inplace=True,
        errors="ignore"
    )

    # -------------------------
    # One-hot encoding
    # -------------------------
    df = pd.get_dummies(
        df,
        columns=["Sex", "Embarked", "Title", "Pclass"],
        drop_first=True
    )

    # -------------------------
    # Feature consistency (CRITICAL)
    # -------------------------
    if is_train:
        feature_columns = sorted(df.columns.tolist())
    else:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

    return df, feature_columns
