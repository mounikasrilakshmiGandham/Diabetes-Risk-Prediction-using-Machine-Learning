"""
eda.py
======
Exploratory Data Analysis for the Pima Indians Diabetes Dataset.

WHY THIS MODULE EXISTS:
  Before training any model, you MUST understand your data.
  This module generates all visualizations needed for:
  - Understanding feature distributions
  - Identifying missing/corrupt values (zeros that should be NaN)
  - Understanding class balance
  - Discovering correlations between features and outcome
  - Detecting outliers

  All plots are saved to the plots/ directory for the research report.

USAGE:
  python -c "from src.eda import run_eda; run_eda('data/diabetes.csv')"
  OR run via train.py which calls this automatically.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend → works on Windows without display
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import ZERO_IS_MISSING, TARGET_COLUMN


# ── Configuration ─────────────────────────────────────────────────
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Consistent color palette
PALETTE = {"Non-Diabetic": "#2196F3", "Diabetic": "#F44336"}
LABEL_MAP = {0: "Non-Diabetic", 1: "Diabetic"}


# ── Internal helpers ──────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path}")
    return path


# ── Public API ────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> str:
    """
    Bar chart showing how many patients are diabetic vs non-diabetic.

    WHY: If classes are highly imbalanced (e.g. 90% non-diabetic),
    the model might predict 'non-diabetic' for everyone and still
    show 90% accuracy — which is misleading. We need to know this upfront.
    """
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = [LABEL_MAP[i] for i in counts.index]
    colors = [PALETTE[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)

    # Annotate bars with count + percentage
    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val}\n({val/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_title("Class Distribution: Diabetic vs Non-Diabetic", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(fig, "01_class_distribution.png")


def plot_feature_distributions(df: pd.DataFrame) -> str:
    """
    Histogram for each feature, colored by class.

    WHY: Shows how feature values differ between diabetic and non-diabetic
    patients. Features with very different distributions are more useful
    for classification (better discriminative power).
    """
    features = [c for c in df.columns if c != TARGET_COLUMN]
    n = len(features)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    df_plot = df.copy()
    df_plot["Label"] = df_plot[TARGET_COLUMN].map(LABEL_MAP)

    for i, feat in enumerate(features):
        ax = axes[i]
        for label, color in PALETTE.items():
            subset = df_plot[df_plot["Label"] == label][feat]
            ax.hist(subset, bins=25, alpha=0.6, color=color,
                    label=label, edgecolor="none")
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    return _save(fig, "02_feature_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """
    Pearson correlation heatmap of all features.

    WHY: Reveals:
    1. Which features are most correlated with Outcome (diabetes target)
       → High absolute correlation = more predictive power
    2. Multi-collinearity between features
       → Highly correlated features are redundant; we may remove one

    For research papers, this heatmap is standard in the methodology section.
    """
    corr_matrix = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # upper triangle mask

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9}
    )

    ax.set_title("Pearson Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()

    return _save(fig, "03_correlation_heatmap.png")


def plot_boxplots(df: pd.DataFrame) -> str:
    """
    Box plots for each feature grouped by diabetes outcome.

    WHY: Box plots show:
    - Median values per class (center line)
    - Spread of data (box = IQR)
    - Outliers (individual dots beyond whiskers)

    Outliers in medical data are critical — a glucose of 500 mg/dL
    is a real medical emergency, NOT a data entry error.
    """
    features = [c for c in df.columns if c != TARGET_COLUMN]
    df_plot = df.copy()
    df_plot["Label"] = df_plot[TARGET_COLUMN].map(LABEL_MAP)

    ncols = 4
    nrows = (len(features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        sns.boxplot(
            data=df_plot,
            x="Label",
            y=feat,
            hue="Label",          # seaborn 0.13+ requires hue for palette
            palette=PALETTE,
            legend=False,          # suppress redundant legend
            ax=ax,
            width=0.5,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        )
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Boxplots by Diabetes Outcome", fontsize=14, fontweight="bold")
    fig.tight_layout()

    return _save(fig, "04_boxplots.png")


def plot_missing_zeros(df: pd.DataFrame) -> str:
    """
    Bar chart showing count of zero-values in columns where
    zero is biologically impossible.

    WHY: The Pima dataset encodes missing values as 0 in columns like
    Glucose, BloodPressure, BMI, etc. A Glucose of 0 means the patient
    is dead — it's clearly a missing value. We must identify these
    before preprocessing, not after.
    """
    zero_counts = {col: (df[col] == 0).sum() for col in ZERO_IS_MISSING}
    zero_counts = {k: v for k, v in zero_counts.items() if v > 0}

    if not zero_counts:
        print("  [INFO] No zero-as-missing values found.")
        return ""

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        list(zero_counts.keys()),
        list(zero_counts.values()),
        color="#FF9800",
        edgecolor="white",
    )
    for bar, val in zip(bars, zero_counts.values()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10)

    ax.set_title("Zero Values (Biologically Impossible → Missing Data)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Count of Zero Values", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return _save(fig, "05_zero_missing_values.png")


def print_statistical_summary(df: pd.DataFrame) -> None:
    """Print descriptive statistics for all features."""
    print("\n" + "=" * 60)
    print("  STATISTICAL SUMMARY")
    print("=" * 60)
    summary = df.describe().round(2)
    print(summary.to_string())

    # Correlation with target
    print("\n  Correlation with Outcome (Pearson):")
    corr_with_target = df.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN)
    corr_sorted = corr_with_target.abs().sort_values(ascending=False)
    for feat, val in corr_sorted.items():
        direction = "↑" if corr_with_target[feat] > 0 else "↓"
        print(f"    {feat:<28}: {val:.4f} {direction}")
    print("=" * 60)


def run_eda(filepath: str) -> pd.DataFrame:
    """
    Run complete EDA pipeline.

    Parameters
    ----------
    filepath : str
        Path to diabetes.csv

    Returns
    -------
    pd.DataFrame
        The loaded dataframe (for chaining into preprocessing).
    """
    from src.data_loader import load_dataset, display_basic_info

    print("\n" + "=" * 60)
    print("  STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = load_dataset(filepath)
    display_basic_info(df)
    print_statistical_summary(df)

    print("\n  Generating plots...")
    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    plot_missing_zeros(df)

    print("\n  [COMPLETE] EDA done. All plots saved to plots/")
    return df


if __name__ == "__main__":
    run_eda(os.path.join("data", "diabetes.csv"))
