"""
feature_selector.py
===================
Implements three complementary feature selection methods:

  Method 1 — Correlation Analysis
    Computes Pearson correlation between each feature and the target.
    Features with |correlation| > threshold are more informative.
    SIMPLE but ignores feature interactions.

  Method 2 — SelectKBest (ANOVA F-test)
    Statistical test: does this feature's value distribution
    differ significantly between diabetic and non-diabetic groups?
    Higher F-score = feature separates classes better.
    STATISTICAL and model-independent.

  Method 3 — Random Forest Feature Importance
    Trains a preliminary Random Forest and measures how much each
    feature reduces impurity (Gini) across all trees.
    CAPTURES non-linear relationships and interactions.

WHY THREE METHODS?
  If a feature ranks high in all three, it's robustly important.
  This triangulation approach is used in published ML research.
  It also defends against the criticism "why did you choose these features?"

RESULT:
  We use all 8 features (the Pima dataset is small — 768 rows).
  But we generate the importance visualizations for the report.
  If the dataset were larger, we would drop low-importance features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier


PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


# ── Method 1: Correlation with Target ────────────────────────────

def correlation_analysis(df: pd.DataFrame, target_col: str = "Outcome") -> pd.Series:
    """
    Pearson correlation between each feature and the outcome.

    Returns sorted Series of absolute correlations.
    """
    corr = df.corr(numeric_only=True)[target_col].drop(target_col)
    abs_corr = corr.abs().sort_values(ascending=False)

    print("\n  [FEATURE SELECTION] Correlation with Outcome:")
    for feat, val in abs_corr.items():
        direction = "positive" if corr[feat] > 0 else "negative"
        print(f"    {feat:<28}: {val:.4f} ({direction})")

    return abs_corr


def plot_correlation_bar(df: pd.DataFrame, target_col: str = "Outcome") -> str:
    """Bar chart of feature correlations with the target."""
    corr = df.corr(numeric_only=True)[target_col].drop(target_col)
    corr_sorted = corr.sort_values(ascending=True)

    colors = ["#F44336" if v > 0 else "#2196F3" for v in corr_sorted.values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(corr_sorted.index, corr_sorted.values,
                   color=colors, edgecolor="white")

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Outcome", fontsize=11)
    ax.set_title("Feature Correlation with Diabetes Outcome\n"
                 "(Red = positive, Blue = negative)", fontsize=12, fontweight="bold")

    for bar, val in zip(bars, corr_sorted.values):
        offset = 0.005 if val >= 0 else -0.005
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha=ha, fontsize=9)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(fig, "06_correlation_with_target.png")


# ── Method 2: SelectKBest (ANOVA F-Test) ─────────────────────────

def selectkbest_analysis(X_train: np.ndarray, y_train: np.ndarray,
                          feature_names: list, k: int = 6) -> list:
    """
    ANOVA F-test: measures if feature distributions differ between classes.

    Parameters
    ----------
    X_train : scaled training features
    y_train : training labels
    feature_names : list of column names
    k : number of top features to select

    Returns
    -------
    selected_features : list of top-k feature names
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    scores = pd.Series(selector.scores_, index=feature_names)
    scores_sorted = scores.sort_values(ascending=False)

    print(f"\n  [FEATURE SELECTION] SelectKBest F-scores (top {k}):")
    for feat, score in scores_sorted.items():
        marker = "★ SELECTED" if feat in scores_sorted.head(k).index else ""
        print(f"    {feat:<28}: F={score:.2f} {marker}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4CAF50" if i < k else "#9E9E9E"
              for i in range(len(scores_sorted))]
    ax.barh(scores_sorted.index[::-1], scores_sorted.values[::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_xlabel("ANOVA F-Score", fontsize=11)
    ax.set_title(f"SelectKBest Feature Scores (Top {k} Selected)",
                 fontsize=12, fontweight="bold")

    # Add score labels
    for i, (feat, score) in enumerate(scores_sorted[::-1].items()):
        ax.text(score + 0.3, i, f"{score:.1f}", va="center", fontsize=9)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "07_selectkbest_scores.png")

    selected = list(scores_sorted.head(k).index)
    return selected


# ── Method 3: Random Forest Feature Importance ───────────────────

def random_forest_importance(X_train: np.ndarray, y_train: np.ndarray,
                               feature_names: list) -> pd.Series:
    """
    Train a quick Random Forest and extract feature importances.

    WHY:
    - RF importance = mean decrease in Gini impurity across all trees
    - Captures non-linear relationships
    - More realistic than correlation (which only captures linear relationships)

    Returns
    -------
    importances : pd.Series sorted by importance descending
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=feature_names)
    importances_sorted = importances.sort_values(ascending=False)

    print("\n  [FEATURE SELECTION] Random Forest Feature Importances:")
    for feat, imp in importances_sorted.items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<28}: {imp:.4f} {bar}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Color gradient: more important = darker green
    n = len(importances_sorted)
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, n))[::-1]

    ax.barh(importances_sorted.index[::-1],
            importances_sorted.values[::-1],
            color=colors[::-1], edgecolor="white")

    ax.set_xlabel("Gini Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title("Random Forest Feature Importance",
                 fontsize=13, fontweight="bold")

    for i, (feat, imp) in enumerate(importances_sorted[::-1].items()):
        ax.text(imp + 0.002, i, f"{imp:.3f}", va="center", fontsize=9)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "08_rf_feature_importance.png")

    return importances_sorted


# ── Summary: Rank features across all methods ─────────────────────

def summarize_feature_selection(df: pd.DataFrame,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  feature_names: list) -> None:
    """
    Run all three methods and print a combined ranking table.
    This is the table you include in your research paper.
    """
    print("\n" + "=" * 60)
    print("  STEP 5: FEATURE SELECTION")
    print("=" * 60)

    # Method 1
    plot_correlation_bar(df)
    corr = df.corr(numeric_only=True)["Outcome"].drop("Outcome").abs()

    # Method 2
    kbest_selected = selectkbest_analysis(X_train, y_train, feature_names)
    selector = SelectKBest(score_func=f_classif, k=len(feature_names))
    selector.fit(X_train, y_train)
    f_scores = pd.Series(selector.scores_, index=feature_names)

    # Method 3
    rf_imp = random_forest_importance(X_train, y_train, feature_names)

    # Normalize all to [0,1] for fair comparison
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    summary = pd.DataFrame({
        "Correlation": normalize(corr),
        "F-Score": normalize(f_scores),
        "RF Importance": normalize(rf_imp),
    })
    summary["Average Rank"] = summary.mean(axis=1)
    summary = summary.sort_values("Average Rank", ascending=False)

    print("\n  Combined Feature Ranking (Normalized 0–1):")
    print(f"  {'Feature':<28} {'Corr':>8} {'F-Score':>9} {'RF Imp':>8} {'Avg':>7}")
    print("  " + "-" * 62)
    for feat, row in summary.iterrows():
        print(f"  {feat:<28} {row['Correlation']:>8.3f} "
              f"{row['F-Score']:>9.3f} {row['RF Importance']:>8.3f} "
              f"{row['Average Rank']:>7.3f}")

    print("\n  → All 8 features retained (small dataset — removing features")
    print("    risks losing predictive signal with only 768 samples)")
    print("\n  [COMPLETE] Feature selection analysis done.")
    print("=" * 60)

    return summary


def run_feature_selection(df: pd.DataFrame, X_train: np.ndarray,
                           y_train: np.ndarray, feature_names: list):
    """Entry point called from train.py."""
    return summarize_feature_selection(df, X_train, y_train, feature_names)
