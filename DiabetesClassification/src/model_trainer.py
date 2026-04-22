"""
model_trainer.py
================
Trains the Random Forest Classifier on the preprocessed Pima dataset.

HYPERPARAMETER DECISIONS (explain each in your viva):

  n_estimators=200
    → Number of decision trees in the forest.
    → More trees = more stable predictions (diminishing returns after ~200
      for this dataset size). 100 would also work, but 200 gives smoother
      probability estimates.

  max_depth=8
    → Limits how deep each tree grows. Prevents overfitting.
    → Without a limit, trees memorize training data perfectly
      (100% train accuracy, poor test accuracy = overfitting).
    → 8 is deep enough to capture non-linear patterns in medical data.

  min_samples_split=5
    → A node is only split further if it has at least 5 samples.
    → Prevents the tree from creating leaves for tiny groups of patients.
    → Reduces overfitting on small subsets of the 768-row dataset.

  min_samples_leaf=2
    → Each leaf must contain at least 2 training samples.
    → Prevents single-patient leaves which would be pure noise.

  class_weight='balanced'
    → Automatically adjusts weights inversely proportional to class frequency.
    → CRITICAL for this dataset: 65% non-diabetic, 35% diabetic.
    → Without this, the model is biased toward predicting "non-diabetic"
      (easier to get high accuracy by just saying "no" to everyone).
    → In medical context, false negatives (missing a diabetic patient)
      are much more costly than false positives. Balanced weights help.

  max_features='sqrt'
    → Each tree only considers sqrt(n_features) features at each split.
    → This is the standard RF approach — reduces correlation between trees,
      making the ensemble more diverse and robust.

  random_state=42
    → Fixed seed ensures results are EXACTLY reproducible.
    → Essential for research: another researcher running your code gets
      identical results. Without this, each run gives slightly different numbers.

  n_jobs=-1
    → Use all available CPU cores during training.
    → On a 4-core machine: 4x faster than n_jobs=1.

  oob_score=True
    → Out-of-bag score: each tree is validated on the samples it didn't
      use during training (bootstrap sample leaves out ~37% of data).
    → Free cross-validation estimate without an extra split.
    → Good for reporting: shows model performance without touching test set.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
RANDOM_STATE = 42
os.makedirs(MODELS_DIR, exist_ok=True)


def build_model() -> RandomForestClassifier:
    """
    Create a Random Forest Classifier with tuned hyperparameters.

    Returns
    -------
    Unfitted RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=200,           # 200 trees
        max_depth=8,                # prevent overfitting
        min_samples_split=5,        # min samples to split a node
        min_samples_leaf=2,         # min samples per leaf
        class_weight="balanced",    # handle class imbalance
        max_features="sqrt",        # standard RF feature subsampling
        random_state=RANDOM_STATE,  # reproducibility
        n_jobs=-1,                  # use all CPU cores
        oob_score=True,             # free OOB validation
        bootstrap=True,             # bootstrap sampling (standard RF)
    )


def cross_validate_model(model: RandomForestClassifier,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          cv: int = 5) -> dict:
    """
    Perform Stratified K-Fold Cross-Validation on training data.

    WHY CROSS-VALIDATION?
    - A single train/test split can be "lucky" (test set happens to be easy).
    - K-Fold: split training data into 5 folds, train on 4, test on 1,
      repeat 5 times → average performance is more reliable.
    - Stratified: each fold maintains original class ratio.

    Parameters
    ----------
    model : RandomForestClassifier (unfitted)
    X_train, y_train : training data
    cv : number of folds (5 is standard)

    Returns
    -------
    dict with mean and std for accuracy, precision, recall, f1
    """
    print(f"\n  Running {cv}-Fold Stratified Cross-Validation...")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    metrics = {}

    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = cross_val_score(
            model, X_train, y_train,
            cv=skf,
            scoring=metric,
            n_jobs=-1
        )
        metrics[metric] = {"mean": scores.mean(), "std": scores.std()}
        print(f"    CV {metric:<12}: {scores.mean():.4f} ± {scores.std():.4f}")

    return metrics


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                run_cv: bool = True) -> RandomForestClassifier:
    """
    Train the Random Forest model.

    Parameters
    ----------
    X_train : np.ndarray — scaled training features
    y_train : np.ndarray — training labels
    run_cv : bool — whether to run cross-validation before final training

    Returns
    -------
    Fitted RandomForestClassifier
    """
    print("\n" + "=" * 60)
    print("  STEP 6: MODEL TRAINING")
    print("=" * 60)
    print(f"  Algorithm    : Random Forest Classifier")
    print(f"  Training set : {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  n_estimators : 200 trees")
    print(f"  max_depth    : 8")
    print(f"  class_weight : balanced (handles 65/35 imbalance)")
    print(f"  random_state : 42 (reproducible)")

    model = build_model()

    # Optional: cross-validation for robust estimate
    if run_cv:
        cv_results = cross_validate_model(model, X_train, y_train)
    else:
        cv_results = {}

    # Final training on full training set
    print(f"\n  Training final model on {X_train.shape[0]} samples...")
    model.fit(X_train, y_train)

    # OOB score (free validation metric)
    print(f"\n  OOB Score (Out-of-Bag): {model.oob_score_:.4f}")
    print(f"  → This is the model's validation accuracy on unseen")
    print(f"    bootstrap samples (no test set contamination).")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\n  [SAVED] Model saved to: {MODEL_PATH}")
    print("  [COMPLETE] Model training done.")
    print("=" * 60)

    return model, cv_results


if __name__ == "__main__":
    # For standalone testing
    print("model_trainer.py: run train.py for full pipeline.")
