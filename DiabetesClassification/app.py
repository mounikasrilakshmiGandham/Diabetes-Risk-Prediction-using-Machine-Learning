"""
app.py
======
Flask Web Application — Diabetes Prediction & Type Classification System.

ROUTES:
  GET  /                      → Render the input form (index.html)
  POST /predict               → Run ML + rule engine, return JSON result
                                 Body param: approach=1 (RF) or approach=2 (Best)
  GET  /api/comparison_results → Return comparison metrics table as JSON
  GET  /api/model_status       → Which models are available on disk
  GET  /health                 → Health check endpoint

APPROACH DESIGN:
  Approach 1 — models/random_forest_model.pkl  (from train.py)
  Approach 2 — models/best_model.pkl           (from train_comparison.py)
  Both share    models/scaler.pkl              (same preprocessing)

USAGE:
  python app.py   →   http://127.0.0.1:5000
"""

import os
import sys
import traceback

from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predictor import (
    predict, validate_input,
    get_model, get_scaler,
    approach2_available, get_best_model_name,
)
from src.model_comparison import load_comparison_results


# ── App ────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ── Pre-load Approach 1 model on startup ──────────────────────────
print("\n[STARTUP] Loading Approach 1 model (Random Forest)...")
try:
    get_model()
    get_scaler()
    print("[STARTUP] Approach 1 loaded successfully.")
except FileNotFoundError as e:
    print(f"\n[ERROR] {e}")
    print("[HINT]  Run 'python train.py' before starting the app.")
    sys.exit(1)

# Approach 2 model is loaded lazily on first use (may not be trained yet)
if approach2_available():
    print(f"[STARTUP] Approach 2 model found: {get_best_model_name()}")
else:
    print("[STARTUP] Approach 2 not found (run train_comparison.py to enable it).")


# ── Routes ─────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Accept patient data + approach selection, return prediction JSON.

    Request body (JSON or form-encoded):
      {
        "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 50,
        "approach": "1"   ← "1" for Random Forest, "2" for Best Model
      }
    """
    if request.is_json:
        raw = dict(request.get_json())
    else:
        raw = request.form.to_dict()

    # Parse and remove approach before validation (it's not a clinical feature)
    try:
        approach = int(raw.pop("approach", 1))
    except (ValueError, TypeError):
        approach = 1

    if approach == 2 and not approach2_available():
        return jsonify({
            "status": "error",
            "message": (
                "Approach 2 model is not available. "
                "Run 'python train_comparison.py' to generate it."
            ),
        }), 400

    # Validate clinical input fields
    cleaned, error = validate_input(raw)
    if error:
        return jsonify({
            "status": "error",
            "message": f"Input validation failed: {error}",
        }), 400

    # Run full prediction pipeline
    try:
        result = predict(cleaned, approach=approach)
        result["status"] = "success"
        return jsonify(result), 200
    except Exception as e:
        print(f"[ERROR] Prediction failed:\n{traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": "Internal error during prediction.",
            "detail": str(e),
        }), 500


@app.route("/api/comparison_results", methods=["GET"])
def comparison_results():
    """
    Return the model comparison metrics table as JSON.
    Used by the UI to render the comparison table without re-running training.
    Returns 404 if train_comparison.py has not been run yet.
    """
    data = load_comparison_results()
    if not data:
        return jsonify({
            "available": False,
            "message": "Run 'python train_comparison.py' to generate comparison results.",
        }), 404
    data["available"] = True
    return jsonify(data), 200


@app.route("/api/model_status", methods=["GET"])
def model_status():
    """Tell the frontend which approaches are ready to use."""
    a2_name = get_best_model_name() if approach2_available() else None
    return jsonify({
        "approach1": {
            "available": True,
            "model_name": "Random Forest",
            "path": "models/random_forest_model.pkl",
        },
        "approach2": {
            "available": approach2_available(),
            "model_name": a2_name,
            "path": "models/best_model.pkl",
        },
    }), 200


@app.route("/health", methods=["GET"])
def health():
    from src.predictor import _model
    return jsonify({
        "status": "ok",
        "approach1_loaded": _model is not None,
        "approach2_available": approach2_available(),
    }), 200


@app.route("/api/features", methods=["GET"])
def features_info():
    return jsonify({
        "features": [
            {"name": "Pregnancies",              "type": "int",   "min": 0,   "max": 20,  "unit": "count"},
            {"name": "Glucose",                  "type": "float", "min": 50,  "max": 300, "unit": "mg/dL"},
            {"name": "BloodPressure",            "type": "float", "min": 40,  "max": 130, "unit": "mm Hg"},
            {"name": "SkinThickness",            "type": "float", "min": 0,   "max": 99,  "unit": "mm"},
            {"name": "Insulin",                  "type": "float", "min": 0,   "max": 900, "unit": "mu U/ml"},
            {"name": "BMI",                      "type": "float", "min": 10,  "max": 70,  "unit": "kg/m²"},
            {"name": "DiabetesPedigreeFunction", "type": "float", "min": 0.0, "max": 2.5, "unit": "score"},
            {"name": "Age",                      "type": "int",   "min": 1,   "max": 120, "unit": "years"},
        ]
    }), 200


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  DIABETES PREDICTION SYSTEM")
    print("  Flask Server Starting...")
    print("=" * 55)
    print("  URL  : http://127.0.0.1:5000")
    print("  Mode : Development (debug=True)")
    print("  Stop : Press Ctrl+C")
    print("=" * 55 + "\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )
