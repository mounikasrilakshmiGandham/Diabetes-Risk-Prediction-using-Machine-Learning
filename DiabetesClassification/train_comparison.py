"""
train_comparison.py
===================
APPROACH 2 — Multi-Model Training Pipeline.

Run this AFTER running train.py (Approach 1), OR independently.
It performs its own preprocessing (same parameters, same random_state)
so the comparison is done on the SAME train/test split as Approach 1.

WHAT THIS SCRIPT DOES:
  1. Load dataset
  2. Preprocess (same steps as train.py — median imputation, scaling)
  3. Run 8-model tournament (model_comparison.py)
  4. Save the winning model as models/best_model.pkl
  5. Save full comparison metrics to models/comparison_results.json
  6. Generate 5 additional comparison plots in plots/

ARTIFACTS PRODUCED:
  models/best_model.pkl            ← winning model (used by Flask Approach 2)
  models/comparison_results.json   ← metrics table for UI display
  plots/12_model_comparison.png
  plots/13_all_roc_curves.png
  plots/14_training_time.png
  plots/15_cv_scores.png
  plots/16_best_model_confusion_matrix.png

USAGE:
  python train_comparison.py

NOTE:
  models/scaler.pkl is shared between Approach 1 and Approach 2.
  Both use the same StandardScaler fitted on the same training data.
  Running this script will regenerate scaler.pkl (identical result
  because random_state=42 guarantees the same split every time).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    start_time = time.time()

    print("\n" + "█" * 65)
    print("  DIABETES CLASSIFICATION — APPROACH 2 TRAINING PIPELINE")
    print("  Multi-Model Comparison & Best Model Selection")
    print("█" * 65)

    # ── Imports ────────────────────────────────────────────────────
    from src.data_loader import load_dataset, display_basic_info
    from src.preprocessor import run_preprocessing
    from src.model_comparison import run_comparison

    # ── Dataset ────────────────────────────────────────────────────
    DATA_PATH = os.path.join("data", "diabetes.csv")
    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Dataset not found: {DATA_PATH}")
        print("  Run: python download_dataset.py")
        sys.exit(1)

    print("\n  Loading dataset...")
    df = load_dataset(DATA_PATH)
    display_basic_info(df)

    # ── Preprocessing (same as train.py — same random_state) ───────
    print("\n  Running preprocessing pipeline...")
    X_train, X_test, y_train, y_test, scaler, feature_names = run_preprocessing(df)

    # ── Multi-model comparison ──────────────────────────────────────
    comparison = run_comparison(X_train, y_train, X_test, y_test)

    # ── Final summary ───────────────────────────────────────────────
    elapsed  = time.time() - start_time
    best     = comparison["best_result"]
    best_name = comparison["best_model_name"]

    print("\n" + "█" * 65)
    print("  APPROACH 2 TRAINING COMPLETE")
    print("█" * 65)
    print(f"  Time elapsed      : {elapsed:.1f} seconds")
    print(f"  Best model        : {best_name}")
    print(f"  Test ROC-AUC      : {best['test_auc']:.4f}")
    print(f"  Test Accuracy     : {best['test_accuracy']*100:.2f}%")
    print(f"  Test Recall       : {best['test_recall']*100:.2f}%")
    print(f"  Test F1-Score     : {best['test_f1']*100:.2f}%")
    print()
    print("  Saved artifacts:")
    print("    models/best_model.pkl")
    print("    models/comparison_results.json")
    print("    plots/12_model_comparison.png")
    print("    plots/13_all_roc_curves.png")
    print("    plots/14_training_time.png")
    print("    plots/15_cv_scores.png")
    print("    plots/16_best_model_confusion_matrix.png")
    print()
    print("  Next: python app.py → select 'Approach 2' in the UI")
    print("█" * 65 + "\n")


if __name__ == "__main__":
    main()
