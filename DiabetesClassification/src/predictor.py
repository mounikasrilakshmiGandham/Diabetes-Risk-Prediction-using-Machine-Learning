"""
predictor.py
============
Inference engine — loads trained model(s) and runs predictions.

DUAL-APPROACH SUPPORT:
  Approach 1 → models/random_forest_model.pkl  (fixed RF, from train.py)
  Approach 2 → models/best_model.pkl           (winner, from train_comparison.py)

  Both approaches share models/scaler.pkl because they use identical
  preprocessing (same dataset, same random_state=42 split).

  The rule engine (Stage 2) is identical for both approaches —
  the ML prediction feeds into the same clinical rule logic.
"""

import os
import json
import numpy as np
import joblib

from src.preprocessor import preprocess_single_input
from src.rule_engine import run_rule_engine


# ── Paths ──────────────────────────────────────────────────────────
MODEL_PATH       = os.path.join("models", "random_forest_model.pkl")
BEST_MODEL_PATH  = os.path.join("models", "best_model.pkl")
SCALER_PATH      = os.path.join("models", "scaler.pkl")
COMPARISON_JSON  = os.path.join("models", "comparison_results.json")


# ── Model cache (loaded once per process) ─────────────────────────

_model      = None   # Approach 1: Random Forest
_best_model = None   # Approach 2: winning model
_scaler     = None


def _load_artifacts():
    """Load Approach 1 model + scaler from disk."""
    global _model, _scaler
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Approach 1 model not found: {MODEL_PATH}\n"
            f"  → Run 'python train.py' first."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Scaler not found: {SCALER_PATH}\n"
            f"  → Run 'python train.py' first."
        )
    _model  = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)
    print(f"  [LOADED] Approach 1 model : {MODEL_PATH}")
    print(f"  [LOADED] Scaler           : {SCALER_PATH}")


def _load_best_model():
    """Load Approach 2 best model from disk (lazy)."""
    global _best_model
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Approach 2 model not found: {BEST_MODEL_PATH}\n"
            f"  → Run 'python train_comparison.py' first."
        )
    _best_model = joblib.load(BEST_MODEL_PATH)
    print(f"  [LOADED] Approach 2 model : {BEST_MODEL_PATH}")


def get_model():
    global _model
    if _model is None:
        _load_artifacts()
    return _model


def get_scaler():
    global _scaler
    if _scaler is None:
        _load_artifacts()
    return _scaler


def get_best_model():
    global _best_model
    if _best_model is None:
        _load_best_model()
    return _best_model


def approach2_available() -> bool:
    """True if the Approach 2 model file exists on disk."""
    return os.path.exists(BEST_MODEL_PATH)


def get_best_model_name() -> str:
    """Read winning model name from the comparison JSON (fast, no pkl load)."""
    if not os.path.exists(COMPARISON_JSON):
        return "Best Model"
    with open(COMPARISON_JSON) as f:
        data = json.load(f)
    return data.get("best_model_name", "Best Model")


# ── Prediction function ────────────────────────────────────────────

def predict(input_dict: dict, approach: int = 1) -> dict:
    """
    End-to-end prediction for a single patient.

    Parameters
    ----------
    input_dict : dict
        Raw clinical values from the web form.
    approach : int
        1 → Approach 1: Random Forest (fixed, from train.py)
        2 → Approach 2: Best model selected by tournament (train_comparison.py)

    Returns
    -------
    dict containing prediction, probability, diabetes type, rules,
    recommendations, approach metadata, and model name.
    """
    # ── Select model based on approach ────────────────────────────
    if approach == 2:
        model = get_best_model()
        model_name    = get_best_model_name()
        approach_label = f"Approach 2 — {model_name} (Auto-Selected)"
    else:
        model = get_model()
        model_name    = "Random Forest"
        approach_label = "Approach 1 — Random Forest (Fixed)"

    # ── Step 1: Preprocess input ───────────────────────────────────
    X_scaled = preprocess_single_input(input_dict, SCALER_PATH)

    # ── Step 2: ML prediction ──────────────────────────────────────
    ml_pred = int(model.predict(X_scaled)[0])            # 0 or 1
    ml_prob = float(model.predict_proba(X_scaled)[0][1]) # P(diabetic)

    # ── Step 3: Rule engine (Stage 2, only if diabetic) ───────────
    rule_result = run_rule_engine(ml_pred, input_dict)

    # ── Step 4: Compose result ─────────────────────────────────────
    result = {
        "prediction":     rule_result["prediction"],
        "probability":    round(ml_prob * 100, 1),
        "probability_raw": ml_prob,
        "input_features": input_dict,
        "approach":       approach,
        "approach_label": approach_label,
        "model_name":     model_name,
    }

    result.update(rule_result)

    # RF/tree ensemble probabilities are compressed (rarely exceed 0.75)
    # due to averaging over 200 trees. Thresholds are set accordingly.
    if ml_prob >= 0.60:
        result["model_confidence"] = "High"
    elif ml_prob >= 0.40:
        result["model_confidence"] = "Moderate"
    else:
        result["model_confidence"] = "Low"

    return result


def validate_input(form_data: dict) -> tuple:
    """
    Validate and convert form input before prediction.

    Returns
    -------
    (cleaned_dict, error_message)
    error_message is None if input is valid.
    """
    FIELDS = {
        "Pregnancies":             (0,   20,   0),
        "Glucose":                 (50,  300,  None),
        "BloodPressure":           (40,  130,  None),
        "SkinThickness":           (0,   99,   0),
        "Insulin":                 (0,   900,  0),
        "BMI":                     (10,  70,   None),
        "DiabetesPedigreeFunction":(0.0, 2.5,  None),
        "Age":                     (1,   120,  None),
    }

    cleaned = {}
    errors = []

    for field, (min_val, max_val, optional_default) in FIELDS.items():
        raw = form_data.get(field, "")

        if raw == "" or raw is None:
            if optional_default is not None:
                cleaned[field] = float(optional_default)
                continue
            else:
                errors.append(f"'{field}' is required.")
                continue

        try:
            val = float(raw)
        except (ValueError, TypeError):
            errors.append(f"'{field}' must be a number (got: '{raw}').")
            continue

        if val < min_val or val > max_val:
            errors.append(
                f"'{field}' = {val} is out of valid range "
                f"[{min_val}–{max_val}]."
            )
            continue

        cleaned[field] = val

    if errors:
        return {}, " | ".join(errors)

    return cleaned, None
