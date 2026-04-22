"""
model_evaluator.py
==================
Evaluates the trained Random Forest model on the held-out test set.

METRICS EXPLAINED (for your viva):

  Accuracy
    → % of all predictions that are correct.
    → MISLEADING alone for imbalanced data.
    → A model that always says "non-diabetic" gets 65% accuracy —
      but is useless medically.

  Precision (for diabetic class)
    → Of all patients predicted as diabetic, what % actually are?
    → Low precision = many false alarms (costly, causes anxiety).

  Recall / Sensitivity (for diabetic class)
    → Of all actual diabetic patients, what % did we catch?
    → Low recall = missed diagnoses (DANGEROUS in medicine).
    → MOST IMPORTANT metric for medical screening.

  F1-Score
    → Harmonic mean of precision and recall.
    → Single number that balances both; best when classes are imbalanced.

  Confusion Matrix
    → Shows TP, TN, FP, FN counts visually.
    → Essential for understanding where the model fails.

  ROC-AUC
    → Area Under the Receiver Operating Characteristic Curve.
    → AUC = 0.5 means random guessing; AUC = 1.0 means perfect.
    → AUC is threshold-independent: shows performance at all cut-offs.
    → Standard metric in medical ML papers.

  MCC (Matthews Correlation Coefficient) — OPTIONAL but impressive
    → Works well for imbalanced classes.
    → Ranges from -1 to +1. +1 = perfect, 0 = random.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier


PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


def compute_metrics(model: RandomForestClassifier,
                     X_test: np.ndarray,
                     y_test: np.ndarray) -> dict:
    """
    Compute all classification metrics on the test set.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc, mcc
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1 (diabetic)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics["auc"] = auc(fpr, tpr)
    metrics["fpr"] = fpr
    metrics["tpr"] = tpr

    return metrics


def print_evaluation_report(metrics: dict, y_test: np.ndarray) -> None:
    """Print a formatted evaluation summary."""
    print("\n" + "=" * 60)
    print("  STEP 7: MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}  ← Most important for medical screening")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['auc']:.4f}")
    print(f"  MCC       : {metrics['mcc']:.4f}")
    print()

    # Interpretation
    auc_val = metrics["auc"]
    if auc_val >= 0.90:
        grade = "Excellent"
    elif auc_val >= 0.80:
        grade = "Good"
    elif auc_val >= 0.70:
        grade = "Fair"
    else:
        grade = "Poor — investigate class imbalance or feature quality"

    print(f"  ROC-AUC Interpretation: {grade}")
    print()

    # Full sklearn report
    print("  Detailed Classification Report:")
    print("  " + "-" * 45)
    report = classification_report(
        y_test, metrics["y_pred"],
        target_names=["Non-Diabetic", "Diabetic"]
    )
    for line in report.strip().split("\n"):
        print(f"  {line}")

    print("=" * 60)


def plot_confusion_matrix(metrics: dict, y_test: np.ndarray) -> str:
    """
    Plot and save the confusion matrix heatmap.

    Matrix layout:
                  Predicted Non-D  Predicted Diabetic
    Actual Non-D       TN                FP
    Actual Diabetic    FN                TP

    FN (False Negative) = most dangerous: missed diabetic patient.
    FP (False Positive) = patient gets follow-up tests unnecessarily.
    """
    cm = confusion_matrix(y_test, metrics["y_pred"])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))

    labels = np.array([
        [f"TN\n{tn}\n(Correct Non-D)", f"FP\n{fp}\n(False Alarm)"],
        [f"FN\n{fn}\n(MISSED!", f"TP\n{tp}\n(Correct Diabetic)"]
    ])

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        ax=ax,
        cbar=True,
        linewidths=2,
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Label", fontsize=12, labelpad=10)
    ax.set_xticklabels(["Non-Diabetic", "Diabetic"], fontsize=10)
    ax.set_yticklabels(["Non-Diabetic", "Diabetic"], fontsize=10, rotation=0)
    ax.set_title("Confusion Matrix — Random Forest\n"
                 f"Accuracy: {(tn+tp)/(tn+fp+fn+tp)*100:.1f}%",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    return _save(fig, "09_confusion_matrix.png")


def plot_roc_curve(metrics: dict) -> str:
    """
    Plot the ROC curve with AUC score.

    WHY ROC-AUC?
    - Shows trade-off between sensitivity (recall) and specificity at all thresholds.
    - The area under the curve summarizes this in one number.
    - Random classifier = 0.50 (diagonal line).
    - Perfect classifier = 1.00.
    - Our target: AUC ≥ 0.80 for a clinically acceptable screening tool.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        metrics["fpr"], metrics["tpr"],
        color="#4CAF50",
        linewidth=2.5,
        label=f"Random Forest (AUC = {metrics['auc']:.4f})"
    )
    ax.plot([0, 1], [0, 1], color="#9E9E9E",
            linestyle="--", linewidth=1.5, label="Random Classifier (AUC = 0.50)")

    # Shade area under curve
    ax.fill_between(metrics["fpr"], metrics["tpr"], alpha=0.1, color="#4CAF50")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_title("ROC Curve — Diabetes Prediction",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, "10_roc_curve.png")


def plot_prediction_probability_distribution(model, X_test: np.ndarray,
                                               y_test: np.ndarray) -> str:
    """
    Distribution of predicted probabilities for each class.

    WHY THIS PLOT?
    - Shows model confidence.
    - A well-calibrated model: non-diabetic patients cluster near 0.0
      and diabetic patients cluster near 1.0.
    - Overlapping distributions = uncertain boundary = harder cases.
    """
    probs = model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, color, name in [(0, "#2196F3", "Non-Diabetic"),
                                 (1, "#F44336", "Diabetic")]:
        subset_probs = probs[y_test == label]
        ax.hist(subset_probs, bins=20, alpha=0.6, color=color,
                label=f"{name} (n={len(subset_probs)})", edgecolor="none")

    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5,
               label="Decision threshold (0.5)")
    ax.set_xlabel("Predicted Probability of Diabetes", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Prediction Probability Distribution by True Class",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(fig, "11_probability_distribution.png")


def run_evaluation(model: RandomForestClassifier,
                    X_test: np.ndarray,
                    y_test: np.ndarray) -> dict:
    """
    Run complete model evaluation pipeline.

    Returns
    -------
    metrics dict
    """
    print("\n" + "=" * 60)
    print("  STEP 7: MODEL EVALUATION")
    print("=" * 60)

    metrics = compute_metrics(model, X_test, y_test)
    print_evaluation_report(metrics, y_test)

    print("\n  Generating evaluation plots...")
    plot_confusion_matrix(metrics, y_test)
    plot_roc_curve(metrics)
    plot_prediction_probability_distribution(model, X_test, y_test)

    print("\n  [COMPLETE] Evaluation done. Plots saved to plots/")
    return metrics
