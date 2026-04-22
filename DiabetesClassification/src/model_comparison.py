"""
model_comparison.py
===================
Approach 2 — Multi-Model Tournament Engine.

PURPOSE:
  Trains 8 different ML classifiers on the SAME preprocessed data,
  evaluates each with 5-fold cross-validation AND on the held-out
  test set, then selects the best model by ROC-AUC score.

WHY RECALL AS SELECTION CRITERION? (Clinical Priority)
  - In medical screening, a False Negative (missed diabetic patient)
    is far more dangerous than a False Positive (unnecessary follow-up).
  - Recall (Sensitivity) = TP / (TP + FN) — directly measures the
    fraction of diabetic patients correctly identified.
  - Maximising Recall minimises missed diagnoses, which prevents
    uncontrolled hyperglycaemia, organ damage, and complications.
  - This follows ADA 2023 screening guidelines: sensitivity is the
    primary criterion for population-level diabetes screening tools.
  - ROC-AUC is still reported (and used as tie-breaker) but Recall
    drives the winner selection.

MODELS EVALUATED:
  1. Logistic Regression   — linear baseline, highly interpretable
  2. Decision Tree         — non-linear, single tree, prone to overfit
  3. Random Forest         — same config as Approach 1 (fair comparison)
  4. Gradient Boosting     — sequential boosting, often top performer
  5. Support Vector Machine— max-margin classifier, powerful in high-dim
  6. K-Nearest Neighbors   — instance-based, no explicit model
  7. Gaussian Naive Bayes  — probabilistic, fastest, assumes independence
  8. Extra Trees           — even more randomized than RF, lower variance

OUTPUTS:
  - models/best_model.pkl            ← winner saved here
  - models/comparison_results.json   ← full metrics table (JSON)
  - plots/12_model_comparison.png    ← grouped bar chart
  - plots/13_all_roc_curves.png      ← all ROC curves on same axes
  - plots/14_training_time.png       ← training time comparison
  - plots/15_cv_scores.png           ← cross-validation scores with error bars
  - plots/16_best_model_cm.png       ← confusion matrix of winner
"""

import os
import json
import time
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix,
    matthews_corrcoef,
)


MODELS_DIR = "models"
PLOTS_DIR  = "plots"
BEST_MODEL_PATH      = os.path.join(MODELS_DIR, "best_model.pkl")
COMPARISON_JSON_PATH = os.path.join(MODELS_DIR, "comparison_results.json")
RANDOM_STATE = 42
CV_FOLDS = 5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  MODEL REGISTRY
#  Each entry: display name → (model_instance, supports_class_weight)
# ══════════════════════════════════════════════════════════════════

def build_model_registry() -> dict:
    """
    Return ordered dict of model name → sklearn estimator.

    Design notes for each model (viva-ready):

    Logistic Regression:
      Fits a sigmoid curve to predict P(diabetic). C=1.0 (default
      regularisation). class_weight='balanced' corrects imbalance.
      Best for: establishing a linear baseline; highly explainable.

    Decision Tree:
      Splits features recursively to minimise Gini impurity.
      max_depth=6 prevents overfitting; min_samples_leaf=5 ensures
      leaves are statistically meaningful.
      Best for: visualising decision boundaries; fully interpretable.

    Random Forest (same as Approach 1):
      200 trees, max_depth=8, class_weight='balanced'.
      This is included so the comparison is FAIR — you can directly
      see if another model outperforms the Approach 1 choice.

    Gradient Boosting:
      Builds trees sequentially; each tree corrects residual errors
      of the previous. n_estimators=200, learning_rate=0.1,
      max_depth=4 (shallow trees work best for boosting).
      NOTE: GBC does not accept class_weight; we pass sample_weight
      at fit() time to handle imbalance.
      Best for: often achieves highest accuracy on tabular data.

    SVM (Support Vector Machine):
      Finds the maximum-margin hyperplane separating classes.
      kernel='rbf' handles non-linear boundaries. C=1.0 (regularisation).
      probability=True enables predict_proba() via Platt scaling.
      class_weight='balanced' handles imbalance.
      Best for: small-to-medium datasets with clear class boundaries.

    K-Nearest Neighbors:
      Predicts by majority vote of K=11 nearest training samples.
      K=11 (odd, >default 5) reduces ties and noise sensitivity.
      weights='distance' gives closer neighbours more influence.
      NOTE: KNN requires feature scaling (already done by StandardScaler).
      Best for: demonstrating that non-parametric methods can compete.

    Gaussian Naive Bayes:
      Applies Bayes theorem assuming features are Gaussian-distributed
      and statistically independent of each other.
      No hyperparameters to tune — very fast baseline.
      The independence assumption is violated in medical data (Glucose
      and Insulin are correlated), so performance may be lower.
      Best for: fastest possible baseline; probabilistic interpretation.

    Extra Trees:
      Like Random Forest but splits are chosen RANDOMLY (not optimally).
      This introduces more variance in trees → lower correlation between
      trees → often lower ensemble variance than RF.
      n_estimators=200, class_weight='balanced'.
      Best for: speed advantage over RF with comparable accuracy.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            oob_score=True,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,            # stochastic GB reduces overfitting
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,         # needed for predict_proba()
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=11,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        ),
        "Naive Bayes": GaussianNB(
            var_smoothing=1e-9,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════

def _needs_sample_weight(name: str) -> bool:
    """GradientBoosting doesn't accept class_weight → use sample_weight."""
    return name == "Gradient Boosting"


def train_and_evaluate_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list:
    """
    Train every model, run cross-validation, evaluate on test set.

    Returns
    -------
    list of dicts, each containing all metrics for one model.
    Sorted by test ROC-AUC descending (best first).
    """
    registry = build_model_registry()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    # Pre-compute sample weights for Gradient Boosting
    gb_sample_weight = compute_sample_weight("balanced", y_train)

    results = []

    print(f"\n  {'Model':<25} {'CV-AUC':>8} {'Test-AUC':>9} "
          f"{'Acc':>7} {'Rec':>7} {'F1':>7} {'Time':>7}")
    print("  " + "-" * 72)

    for name, model in registry.items():
        # ── Cross-validation (on training set) ─────────────────
        cv_auc_scores = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring="roc_auc", n_jobs=-1
        )
        cv_auc_mean = cv_auc_scores.mean()
        cv_auc_std  = cv_auc_scores.std()

        cv_f1_scores = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring="f1", n_jobs=-1
        )

        # ── Train on full training set ──────────────────────────
        t0 = time.perf_counter()
        if _needs_sample_weight(name):
            model.fit(X_train, y_train, sample_weight=gb_sample_weight)
        else:
            model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # ── Evaluate on held-out test set ───────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        test_auc = auc(fpr, tpr)

        record = {
            "name":          name,
            "cv_auc_mean":   round(float(cv_auc_mean), 4),
            "cv_auc_std":    round(float(cv_auc_std),  4),
            "cv_f1_mean":    round(float(cv_f1_scores.mean()), 4),
            "test_auc":      round(float(test_auc),    4),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "test_precision":round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "test_recall":   round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "test_f1":       round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "test_mcc":      round(float(matthews_corrcoef(y_test, y_pred)), 4),
            "train_time_s":  round(float(train_time), 3),
            "fpr":           fpr.tolist(),
            "tpr":           tpr.tolist(),
            "fitted_model":  model,          # held in memory, not serialised to JSON
        }

        results.append(record)
        print(f"  {name:<25} {cv_auc_mean:>8.4f} {test_auc:>9.4f} "
              f"{record['test_accuracy']:>7.4f} {record['test_recall']:>7.4f} "
              f"{record['test_f1']:>7.4f} {train_time:>6.3f}s")

    # Sort by test Recall descending (clinical priority: minimise missed diagnoses)
    # Tie-breaker: test_auc
    results.sort(key=lambda r: (r["test_recall"], r["test_auc"]), reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


def plot_model_comparison_bar(results: list) -> str:
    """
    Grouped horizontal bar chart comparing all models across 5 metrics.
    Best model is highlighted in a distinct colour.
    """
    metrics = ["test_accuracy", "test_precision", "test_recall",
               "test_f1", "test_auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    names   = [r["name"] for r in results]
    best    = results[0]["name"]      # already sorted by AUC desc

    n_models  = len(names)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.14
    offsets = np.linspace(-(n_metrics-1)/2, (n_metrics-1)/2, n_metrics) * width

    PALETTE = ["#1565C0", "#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, PALETTE)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + offsets[i], vals, width=width,
                      label=label, color=color, alpha=0.85, edgecolor="white")

        # Annotate top bar value (only for AUC to avoid clutter)
        if metric == "test_auc":
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        fontsize=7, color="#1565C0", fontweight="bold")

    # Highlight best model x-tick
    ax.set_xticks(x)
    xticklabels = ax.set_xticklabels(
        [f"★ {n}" if n == best else n for n in names],
        rotation=20, ha="right", fontsize=9
    )
    for lbl, name in zip(ax.get_xticklabels(), names):
        if name == best:
            lbl.set_fontweight("bold")
            lbl.set_color("#1565C0")

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Approach 2 — Model Comparison (All Metrics)\n"
                 f"★ Best by Recall (Clinical): {best}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(fig, "12_model_comparison.png")


def plot_all_roc_curves(results: list, y_test: np.ndarray) -> str:
    """
    All 8 ROC curves on the same axes so they can be directly compared.
    The best model's curve is drawn thicker and in a distinct colour.
    """
    COLORS = [
        "#E91E63", "#9C27B0", "#1565C0", "#0097A7",
        "#2E7D32", "#F57C00", "#795548", "#607D8B",
    ]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Random baseline
    ax.plot([0, 1], [0, 1], color="#BDBDBD", linestyle="--",
            linewidth=1.2, label="Random (AUC = 0.50)", zorder=1)

    best_name = results[0]["name"]

    for i, r in enumerate(results):
        is_best = (r["name"] == best_name)
        lw    = 3.0 if is_best else 1.4
        alpha = 1.0 if is_best else 0.7
        ls    = "-"
        zorder = 10 if is_best else 2
        label = (f"★ {r['name']} (AUC={r['test_auc']:.3f})"
                 if is_best else
                 f"{r['name']} (AUC={r['test_auc']:.3f})")

        ax.plot(r["fpr"], r["tpr"],
                color=COLORS[i % len(COLORS)],
                linewidth=lw, alpha=alpha,
                linestyle=ls, label=label, zorder=zorder)

    # Shade under best curve
    ax.fill_between(results[0]["fpr"], results[0]["tpr"],
                    alpha=0.07, color=COLORS[0])

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Approach 2 — ROC Curves: All Models",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(fig, "13_all_roc_curves.png")


def plot_training_time(results: list) -> str:
    """
    Horizontal bar chart of training time for each model.
    Useful for demonstrating the speed-accuracy trade-off in the report.
    """
    names = [r["name"] for r in results]
    times = [r["train_time_s"] for r in results]
    best  = results[0]["name"]
    colors = ["#1565C0" if n == best else "#90CAF9" for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, times, color=colors, edgecolor="white")

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max(times)*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{t:.3f}s", va="center", fontsize=9)

    ax.set_xlabel("Training Time (seconds)", fontsize=11)
    ax.set_title("Model Training Time Comparison\n"
                 "(Measured on same hardware, same dataset)",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(fig, "14_training_time.png")


def plot_cv_scores(results: list) -> str:
    """
    Cross-validation ROC-AUC scores with error bars (mean ± std).
    Error bars show model stability — a small std = reliable performance.
    """
    names    = [r["name"] for r in results]
    means    = [r["cv_auc_mean"] for r in results]
    stds     = [r["cv_auc_std"]  for r in results]
    best     = results[0]["name"]
    colors   = ["#1565C0" if n == best else "#90CAF9" for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))

    bars = ax.bar(x, means, color=colors, edgecolor="white", width=0.55)
    ax.errorbar(x, means, yerr=stds,
                fmt="none", color="#212121",
                capsize=5, capthick=1.5, elinewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"★ {n}" if n == best else n for n in names],
        rotation=20, ha="right", fontsize=9
    )

    # Annotate mean values
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + std + 0.008,
                f"{mean:.3f}", ha="center",
                va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("5-Fold CV ROC-AUC (mean ± std)", fontsize=11)
    ax.set_title("Cross-Validation Stability — ROC-AUC per Model\n"
                 "(Error bars = standard deviation across 5 folds)",
                 fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(fig, "15_cv_scores.png")


def plot_best_model_confusion_matrix(best_result: dict,
                                      y_test: np.ndarray) -> str:
    """Confusion matrix for the winning model (identical format to Approach 1)."""
    model = best_result["fitted_model"]
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(
        # We need X_test — passed via closure; see run_comparison()
        _stored_X_test
    )
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    labels = np.array([
        [f"TN\n{tn}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TP\n{tp}"],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt="",
                cmap="Blues", ax=ax, linewidths=2, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xticklabels(["Non-Diabetic", "Diabetic"], fontsize=10)
    ax.set_yticklabels(["Non-Diabetic", "Diabetic"], fontsize=10, rotation=0)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_title(f"Best Model Confusion Matrix\n"
                 f"{best_result['name']}  |  "
                 f"Accuracy: {(tn+tp)/(tn+fp+fn+tp)*100:.1f}%",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "16_best_model_confusion_matrix.png")


# Module-level store so plot fn can access X_test without signature change
_stored_X_test: np.ndarray = None


# ══════════════════════════════════════════════════════════════════
#  SAVE / LOAD RESULTS
# ══════════════════════════════════════════════════════════════════

def save_comparison_results(results: list, best_name: str) -> None:
    """
    Persist the comparison metrics table to JSON.
    The Flask API serves this to the UI so the browser can render
    the comparison table without re-running training.
    """
    # Strip non-serialisable fields before JSON dump
    serialisable = []
    for r in results:
        row = {k: v for k, v in r.items()
               if k not in ("fitted_model", "fpr", "tpr")}
        row["is_best"] = (r["name"] == best_name)
        serialisable.append(row)

    payload = {
        "best_model_name": best_name,
        "selection_criterion": "test_recall",
        "cv_folds": CV_FOLDS,
        "n_models": len(results),
        "models": serialisable,
    }

    with open(COMPARISON_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  [SAVED] {COMPARISON_JSON_PATH}")


def load_comparison_results() -> dict:
    """Load the JSON comparison table (used by Flask API)."""
    if not os.path.exists(COMPARISON_JSON_PATH):
        return {}
    with open(COMPARISON_JSON_PATH) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════
#  MASTER ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def run_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Run the full multi-model comparison pipeline.

    Steps:
    1. Train all 8 models + cross-validate
    2. Rank by test ROC-AUC
    3. Generate all comparison plots
    4. Save best model as best_model.pkl
    5. Save metrics table as comparison_results.json

    Parameters
    ----------
    X_train, y_train : scaled training data (from preprocessor)
    X_test, y_test   : scaled test data (from preprocessor)

    Returns
    -------
    dict with keys: best_model_name, best_result, all_results
    """
    global _stored_X_test
    _stored_X_test = X_test

    print("\n" + "=" * 72)
    print("  APPROACH 2 — MULTI-MODEL COMPARISON")
    print("=" * 72)
    print(f"  Models evaluated : 8")
    print(f"  CV folds         : {CV_FOLDS}-Fold Stratified")
    print(f"  Selection metric : Test Recall (Clinical — minimise missed diagnoses)")
    print(f"  Train set size   : {X_train.shape[0]} samples")
    print(f"  Test set size    : {X_test.shape[0]} samples")
    print()

    # 1. Train + evaluate all models
    results = train_and_evaluate_all(X_train, y_train, X_test, y_test)

    best_result = results[0]
    best_name   = best_result["name"]

    # 2. Print ranked leaderboard
    print("\n  LEADERBOARD (ranked by Test Recall [Clinical Priority]):")
    print(f"  {'Rank':<5} {'Model':<25} {'CV-AUC':>9} {'Test-AUC':>9} "
          f"{'Recall':>8} {'F1':>7}")
    print("  " + "-" * 63)
    for rank, r in enumerate(results, 1):
        marker = "★ WINNER" if rank == 1 else ""
        print(f"  {rank:<5} {r['name']:<25} {r['cv_auc_mean']:>9.4f} "
              f"{r['test_auc']:>9.4f} {r['test_recall']:>8.4f} "
              f"{r['test_f1']:>7.4f}  {marker}")

    print(f"\n  Best model (clinical): {best_name}")
    print(f"  Test Recall:  {best_result['test_recall']:.4f}  ← selection criterion")
    print(f"  Test ROC-AUC: {best_result['test_auc']:.4f}  ← tie-breaker")

    # 3. Plots
    print("\n  Generating comparison plots...")
    plot_model_comparison_bar(results)
    plot_all_roc_curves(results, y_test)
    plot_training_time(results)
    plot_cv_scores(results)
    plot_best_model_confusion_matrix(best_result, y_test)

    # 4. Save best model
    joblib.dump(best_result["fitted_model"], BEST_MODEL_PATH)
    print(f"  [SAVED] Best model → {BEST_MODEL_PATH}")

    # 5. Save comparison JSON (strips non-serialisable fields)
    save_comparison_results(results, best_name)

    print("\n  [COMPLETE] Multi-model comparison done.")
    print("=" * 72)

    return {
        "best_model_name": best_name,
        "best_result":     best_result,
        "all_results":     results,
    }
