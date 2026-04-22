"""
data_loader.py
==============
Responsible for loading and performing initial validation of the
Pima Indians Diabetes Dataset.

WHY THIS MODULE EXISTS:
  Loading data seems trivial, but doing it correctly means:
  - Verifying the file exists
  - Confirming expected columns are present
  - Reporting basic shape and class distribution
  This makes the pipeline robust and debuggable.
"""

import pandas as pd
import numpy as np
import os
import sys


# ── Column definitions ────────────────────────────────────────────
EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

TARGET_COLUMN = "Outcome"

# Columns where 0 is biologically impossible → treat as missing
ZERO_IS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the Pima Indians Diabetes CSV file.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to diabetes.csv

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {filepath}\n"
            f"  → Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
            f"  → Save as: data/diabetes.csv"
        )

    df = pd.read_csv(filepath)

    # Validate columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[ERROR] Missing columns in dataset: {missing_cols}\n"
            f"  Expected: {EXPECTED_COLUMNS}\n"
            f"  Found:    {list(df.columns)}"
        )

    return df


def display_basic_info(df: pd.DataFrame) -> None:
    """Print a structured summary of the loaded dataset."""
    print("=" * 60)
    print("  DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Target column  : {TARGET_COLUMN}")
    print(f"  Missing values : {df.isnull().sum().sum()} explicit NaN")
    print()

    # Class distribution
    counts = df[TARGET_COLUMN].value_counts()
    total = len(df)
    print("  Class Distribution:")
    print(f"    Non-Diabetic (0) : {counts.get(0, 0):>4}  "
          f"({counts.get(0, 0)/total*100:.1f}%)")
    print(f"    Diabetic     (1) : {counts.get(1, 0):>4}  "
          f"({counts.get(1, 0)/total*100:.1f}%)")
    print()

    # Zero-as-missing counts
    print("  Biological Zero Values (treated as missing):")
    for col in ZERO_IS_MISSING:
        n_zeros = (df[col] == 0).sum()
        print(f"    {col:<28}: {n_zeros} zeros ({n_zeros/total*100:.1f}%)")
    print("=" * 60)


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (all columns except Outcome)."""
    return [c for c in df.columns if c != TARGET_COLUMN]


if __name__ == "__main__":
    # Quick sanity test
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")
    df = load_dataset(data_path)
    display_basic_info(df)
    print(df.head())
