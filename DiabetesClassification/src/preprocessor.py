"""
preprocessor.py
===============
Handles all data cleaning, transformation, and splitting steps.

WHY THIS MODULE EXISTS:
  Raw data cannot go directly into an ML model. This module:
  1. Replaces biologically impossible zeros with NaN
  2. Imputes missing values using the MEDIAN (not mean — median is
     robust to outliers, which medical data often has)
  3. Scales features using StandardScaler so all features are on
     the same numeric scale (important for some algorithms, and
     for fair comparison of feature importance)
  4. Splits data into train/test sets with stratification
     (stratification ensures both sets have the same class ratio)
  5. Saves the fitted scaler so inference uses the SAME scaling

IMPORTANT DESIGN DECISION:
  The scaler is fit ONLY on training data, then applied to both
  train and test sets. Fitting on the full dataset would be
  "data leakage" — the model would indirectly see test data during
  training, inflating performance metrics artificially.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.data_loader import ZERO_IS_MISSING, TARGET_COLUMN


# ── Constants ──────────────────────────────────────────────────────
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
RANDOM_STATE = 42          # Fixed seed → reproducible splits every run
TEST_SIZE = 0.20           # 80% train, 20% test
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Step 1: Replace impossible zeros with NaN ──────────────────────

def replace_zero_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    In the Pima dataset, certain columns use 0 to represent missing data.
    Glucose = 0, BloodPressure = 0, BMI = 0 are medically impossible values.
    We replace them with NaN so the imputer handles them correctly.

    Columns treated this way: Glucose, BloodPressure,
                               SkinThickness, Insulin, BMI
    """
    df = df.copy()
    for col in ZERO_IS_MISSING:
        n_replaced = (df[col] == 0).sum()
        if n_replaced > 0:
            df[col] = df[col].replace(0, np.nan)
            print(f"  [IMPUTE] Replaced {n_replaced} zeros in '{col}' with NaN")
    return df


# ── Step 2: Impute missing values ─────────────────────────────────

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values with the MEDIAN of that column.

    WHY MEDIAN, not MEAN?
    - Medical data often has outliers (e.g., Insulin can range 0–846)
    - Mean is pulled by extreme values; Median is robust to them
    - This is a standard medical data imputation technique

    WHY NOT DROP ROWS?
    - Dropping reduces the dataset from 768 to potentially 392 rows
    - Losing 376 real patient records weakens the model significantly
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]

    imputer = SimpleImputer(strategy="median")
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    remaining_nan = df.isnull().sum().sum()
    print(f"  [IMPUTE] Imputation complete. Remaining NaN: {remaining_nan}")
    return df


# ── Step 3: Split into train/test ──────────────────────────────────

def split_data(df: pd.DataFrame):
    """
    Split dataframe into training and test sets.

    WHY STRATIFY?
    - Without stratification, random splits might put all diabetic
      patients in training and none in test (or vice versa)
    - Stratify=y ensures both sets maintain the original 65/35 ratio

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    """
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y            # ← crucial for imbalanced datasets
    )

    print(f"  [SPLIT] Train: {X_train.shape[0]} samples | "
          f"Test: {X_test.shape[0]} samples")
    print(f"  [SPLIT] Train diabetic rate : "
          f"{y_train.mean()*100:.1f}%")
    print(f"  [SPLIT] Test  diabetic rate : "
          f"{y_test.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test


# ── Step 4: Feature scaling ────────────────────────────────────────

def scale_features(X_train: np.ndarray, X_test: np.ndarray, save: bool = True):
    """
    Apply StandardScaler: transforms each feature to mean=0, std=1.

    WHY SCALE?
    - Random Forest doesn't strictly require scaling
    - But StandardScaler helps when using SelectKBest (chi2 needs non-negative,
      but we use f_classif which works on standardized values)
    - Ensures the saved scaler can standardize user input at inference time
      (same transformation applied to new data as was applied to training data)

    WHY FIT ONLY ON TRAINING DATA?
    - The scaler "learns" mean and std from training data
    - Applying it to test data simulates "the model has never seen test data"
    - This is a fundamental rule of ML — no data leakage

    Parameters
    ----------
    X_train, X_test : numpy arrays
    save : bool
        If True, saves scaler.pkl for inference use

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)    # fit + transform train
    X_test_scaled = scaler.transform(X_test)           # transform ONLY (no fit)

    if save:
        joblib.dump(scaler, SCALER_PATH)
        print(f"  [SAVED] Scaler saved to: {SCALER_PATH}")

    return X_train_scaled, X_test_scaled, scaler


# ── Master preprocessing function ─────────────────────────────────

def run_preprocessing(df: pd.DataFrame):
    """
    Run the complete preprocessing pipeline.

    Steps:
    1. Replace biological zeros with NaN
    2. Impute with median
    3. Train/test split (stratified)
    4. Scale features (fit on train only)

    Returns
    -------
    X_train, X_test, y_train, y_test : scaled numpy arrays
    scaler : fitted StandardScaler (also saved to disk)
    feature_names : list of feature column names
    """
    print("\n" + "=" * 60)
    print("  STEP 4: DATA PREPROCESSING")
    print("=" * 60)

    feature_names = [c for c in df.columns if c != TARGET_COLUMN]

    df = replace_zero_with_nan(df)
    df = impute_missing_values(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test, save=True)

    print("\n  [COMPLETE] Preprocessing done.")
    print(f"  Features used: {feature_names}")

    return X_train_s, X_test_s, y_train, y_test, scaler, feature_names


# ── Inference helper (used by Flask API) ──────────────────────────

def preprocess_single_input(input_dict: dict, scaler_path: str = SCALER_PATH) -> np.ndarray:
    """
    Preprocess a single patient input for inference.

    This function applies the SAME transformations that were applied
    during training — using the SAVED scaler (not a new one).

    Parameters
    ----------
    input_dict : dict
        Keys must match feature names exactly:
        {Pregnancies, Glucose, BloodPressure, SkinThickness,
         Insulin, BMI, DiabetesPedigreeFunction, Age}
    scaler_path : str
        Path to saved scaler.pkl

    Returns
    -------
    np.ndarray of shape (1, n_features) — ready for model.predict()
    """
    FEATURE_ORDER = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    # Build array in the correct column order
    values = []
    for feat in FEATURE_ORDER:
        val = input_dict.get(feat, 0)
        # Replace biological zeros with median defaults if zero
        if feat in ZERO_IS_MISSING and val == 0:
            val = np.nan
        values.append(float(val))

    arr = np.array(values).reshape(1, -1)

    # Handle NaN (shouldn't happen from form input, but be safe)
    nan_indices = np.where(np.isnan(arr))
    if len(nan_indices[0]) > 0:
        # Use conservative defaults for NaN values
        defaults = {
            "Glucose": 117.0,
            "BloodPressure": 72.0,
            "SkinThickness": 23.0,
            "Insulin": 30.5,
            "BMI": 32.0,
        }
        for col_idx in nan_indices[1]:
            feat = FEATURE_ORDER[col_idx]
            arr[0, col_idx] = defaults.get(feat, 0.0)

    # Load and apply saved scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"[ERROR] Scaler not found at {scaler_path}. "
            f"Run train.py first to generate it."
        )

    scaler = joblib.load(scaler_path)
    arr_scaled = scaler.transform(arr)

    return arr_scaled
