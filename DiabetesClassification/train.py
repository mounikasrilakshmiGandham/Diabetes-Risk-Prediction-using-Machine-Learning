"""
train.py
========
MASTER TRAINING PIPELINE — Run this ONCE to train and save the model.

This script orchestrates all training steps in order:
  1. Load dataset
  2. Exploratory Data Analysis (EDA)
  3. Preprocessing (clean, impute, split, scale)
  4. Feature selection analysis
  5. Train Random Forest
  6. Evaluate model
  7. Save model + scaler

After this script completes:
  - models/random_forest_model.pkl  ← trained model
  - models/scaler.pkl               ← fitted scaler
  - plots/*.png                     ← all visualizations

USAGE:
  python train.py

IMPORTANT:
  You must run this BEFORE starting the Flask app.
  The app loads the saved .pkl files — it does NOT retrain on each request.
  Training happens OFFLINE; inference happens ONLINE via Flask.
"""

import os
import sys
import time

# Ensure src/ is importable regardless of where you run train.py from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    start_time = time.time()

    print("\n" + "█" * 60)
    print("  DIABETES CLASSIFICATION SYSTEM — TRAINING PIPELINE")
    print("█" * 60)
    print(f"  Working directory: {os.getcwd()}")

    # ── Step 1: Import modules ─────────────────────────────────────
    from src.data_loader import load_dataset
    from src.eda import run_eda
    from src.preprocessor import run_preprocessing
    from src.feature_selector import run_feature_selection
    from src.model_trainer import train_model
    from src.model_evaluator import run_evaluation

    # ── Step 2: Dataset path ───────────────────────────────────────
    DATA_PATH = os.path.join("data", "diabetes.csv")

    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Dataset not found at: {DATA_PATH}")
        print("  Run this first: python download_dataset.py")
        sys.exit(1)

    # ── Step 3: EDA ────────────────────────────────────────────────
    df = run_eda(DATA_PATH)

    # ── Step 4: Preprocessing ─────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, feature_names = run_preprocessing(df)

    # ── Step 5: Feature Selection ─────────────────────────────────
    feature_summary = run_feature_selection(df, X_train, y_train, feature_names)

    # ── Step 6: Train Model ────────────────────────────────────────
    model, cv_results = train_model(X_train, y_train, run_cv=True)

    # ── Step 7: Evaluate Model ────────────────────────────────────
    metrics = run_evaluation(model, X_test, y_test)

    # ── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    print("\n" + "█" * 60)
    print("  TRAINING COMPLETE")
    print("█" * 60)
    print(f"  Time elapsed     : {elapsed:.1f} seconds")
    print(f"  Model saved to   : models/random_forest_model.pkl")
    print(f"  Scaler saved to  : models/scaler.pkl")
    print(f"  Plots saved to   : plots/")
    print()
    print("  Final Test Set Metrics:")
    print(f"    Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"    Recall    : {metrics['recall']*100:.2f}%")
    print(f"    Precision : {metrics['precision']*100:.2f}%")
    print(f"    F1-Score  : {metrics['f1']*100:.2f}%")
    print(f"    ROC-AUC   : {metrics['auc']:.4f}")
    print()
    print("  Next step: run 'python app.py' to start the web interface")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
