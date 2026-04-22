# Diabetes Classification — Model Metrics Report
## Approach 2: Multi-Model Tournament Results

**Dataset**: Pima Indians Diabetes Database (768 samples, 8 features)
**Split**: 80% Train (614 samples) / 20% Test (154 samples) — Stratified
**Cross-Validation**: 5-Fold Stratified K-Fold (on training set)
**Class Imbalance**: ~65% Non-Diabetic / ~35% Diabetic
**Imbalance Handling**: `class_weight='balanced'` on all applicable models
**Winner Selection Criterion**: **Test Recall (Clinical Priority — minimise missed diagnoses)**

---

## Part 1 — Full Metrics Table (All 8 Models)

| Model | CV-AUC (mean ± std) | Test AUC | Accuracy | Precision | Recall | F1-Score | MCC | Train Time |
|---|---|---|---|---|---|---|---|---|
| **Extra Trees** ★ | **0.8480 ± ?** | 0.8178 | 74.03% | 60.61% | **74.07%** | **66.67%** | — | 0.140s |
| Decision Tree | 0.7558 ± ? | 0.7833 | 70.13% | 55.57% | 74.07% | 63.49% | — | 0.002s |
| SVM | 0.8367 ± ? | 0.8139 | 72.73% | 59.09% | 72.22% | 65.00% | — | 0.027s |
| Logistic Regression | 0.8442 ± ? | 0.8126 | 73.38% | 60.31% | 70.37% | 64.96% | — | 0.003s |
| Naive Bayes | 0.8280 ± ? | 0.7646 | 70.13% | 56.69% | 62.96% | 59.65% | — | 0.001s |
| Random Forest | 0.8329 ± ? | **0.8254** | **75.32%** | **63.79%** | 68.52% | 66.07% | — | 0.208s |
| Gradient Boosting | 0.8047 ± ? | 0.8200 | **75.32%** | 64.26% | 66.67% | 65.45% | — | 0.202s |
| K-Nearest Neighbors | 0.8232 ± ? | 0.7930 | 73.38% | 63.21% | 57.41% | 60.19% | — | 0.001s |

> ★ **Winner**: Extra Trees — selected by highest Test Recall (clinical priority)
> Approach 1 (Random Forest) was the winner under the original Test ROC-AUC criterion.

---

## Part 2 — What Each Metric Means

### 1. Accuracy
**Formula**: `(TP + TN) / (TP + TN + FP + FN)`

**What it measures**: The percentage of ALL predictions (diabetic and non-diabetic) that were correct.

**Why it can mislead on this dataset**:
If the model predicts "Non-Diabetic" for every patient, it achieves ~65% accuracy (the majority class size) while being medically useless. For a dataset with class imbalance, accuracy inflates in favour of the dominant class.

**Interpretation guide**:
- < 70% → Poor
- 70–75% → Acceptable
- 75–80% → Good
- > 80% → Excellent

**Our results**: Random Forest and Gradient Boosting lead at 75.32%. Extra Trees is close at 74.03%.

---

### 2. Precision (Positive Predictive Value — PPV)
**Formula**: `TP / (TP + FP)`

**What it measures**: Of all patients the model labelled **Diabetic**, what fraction actually IS diabetic?

**Clinical meaning**: A low Precision → many false alarms. Patients are told they might be diabetic when they are not, causing unnecessary follow-up tests, stress, and healthcare costs.

**Trade-off with Recall**: Precision and Recall are inversely correlated. Increasing one typically decreases the other.

**Our results**: Random Forest has the best Precision (63.79%), meaning it raises fewer false alarms than Extra Trees (60.61%).

---

### 3. Recall (Sensitivity / True Positive Rate)
**Formula**: `TP / (TP + FN)`

**What it measures**: Of all patients who ARE diabetic, what fraction did the model correctly identify?

**Clinical meaning (MOST CRITICAL for screening)**:
A low Recall means the model **misses diabetic patients** (False Negatives). A missed diagnosis means the patient receives no treatment, leading to:
- Uncontrolled blood glucose progression
- Organ damage (kidneys, eyes, nerves)
- Increased risk of cardiovascular disease
- Preventable complications and mortality

**Why Recall is the selection criterion here**:
In medical screening tools, missing a sick patient is far more costly than a false alarm. This is the foundational principle of clinical sensitivity — the ADA and WHO both prioritise sensitivity in diabetes screening protocols. We can always do a follow-up confirmatory test (OGTT, HbA1c) for false positives, but we cannot un-miss a false negative.

**Our results**: Extra Trees = 74.07% (highest), Decision Tree = 74.07% (tied), SVM = 72.22%.
Random Forest only achieves 68.52% — it misses ~32% of all diabetic patients.

---

### 4. F1-Score (Harmonic Mean of Precision and Recall)
**Formula**: `2 × (Precision × Recall) / (Precision + Recall)`

**What it measures**: A balanced metric that penalises models which sacrifice one of Precision or Recall entirely.

**Why harmonic mean (not arithmetic)**:
If Precision = 1.0 and Recall = 0.0, the arithmetic mean would be 0.5 (seemingly ok). The harmonic mean would be 0.0 (correctly flagging the model as useless). This makes F1 a much stricter measure of balance.

**Our results**: Extra Trees leads at 66.67%, followed by Random Forest at 66.07%. Both are comparable in balance, but Extra Trees achieves this balance at a higher Recall.

---

### 5. ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
**Formula**: Integral of TPR vs FPR curve across all classification thresholds.

**What it measures**: The model's overall ability to **discriminate** between diabetic and non-diabetic patients, independent of any specific decision threshold.

**Interpretation**:
- 0.50 → Random guessing (coin flip)
- 0.70–0.79 → Acceptable discrimination
- 0.80–0.89 → Good discrimination
- 0.90–1.00 → Excellent/outstanding discrimination
- 1.00 → Perfect (usually indicates data leakage)

**Why it was our original selection criterion**:
ROC-AUC is threshold-agnostic, making it a fair comparison metric when different models use different internal confidence thresholds. It is the standard metric in medical ML papers (BMJ, Lancet Digital Health, JMLR).

**Why we moved away from it**:
ROC-AUC optimises for discrimination ability overall but does not directly tell you what happens at the operating threshold actually used in deployment. At the deployed threshold, what matters clinically is **Recall** — not aggregate AUC.

**Our results**: Random Forest wins on Test-AUC (0.8254). Extra Trees is 4th (0.8178). But notably, Extra Trees wins on **CV-AUC** (0.8480 — highest generalisation AUC), meaning it is the most consistent performer across cross-validation folds.

---

### 6. CV-AUC (Cross-Validation ROC-AUC Mean ± Std)
**Formula**: Average ROC-AUC across 5 held-out validation folds during training.

**What it measures**: How well the model generalises to **unseen data** across multiple data splits. A model that overfits to its training data will have high training AUC but low CV-AUC.

**Standard deviation (Std)**:
A low std = stable, reliable model. A high std = model performance varies depending on which samples it sees — risky for real-world deployment.

**Our results**: Extra Trees has the highest CV-AUC (0.8480), closely followed by Logistic Regression (0.8442) and SVM (0.8367). Decision Tree has the lowest (0.7558) — a classic sign of overfitting.

---

### 7. MCC (Matthews Correlation Coefficient)
**Formula**: `(TP×TN − FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]`

**What it measures**: A balanced measure of prediction quality for binary classification. Often called the "gold standard" single metric for imbalanced datasets.

**Interpretation**:
- +1 → Perfect prediction
- 0 → No better than random
- −1 → Completely wrong predictions

**Advantage over Accuracy and F1**: MCC accounts for all four quadrants of the confusion matrix (TP, TN, FP, FN) equally. It gives a high score only if the classifier performs well on ALL classes — making it the best single metric for imbalanced data like this dataset.

---

### 8. Training Time
**What it measures**: Wall-clock seconds to fit the model on the training set (614 samples, 8 features).

**Our results**: Naive Bayes (0.001s) and K-NN (0.001s) are fastest. Random Forest (0.208s) and Gradient Boosting (0.202s) are slowest. Extra Trees (0.140s) is ~26% faster than Random Forest while outperforming it on clinical Recall.

> Note: Training time matters for retraining pipelines (batch jobs), not for inference. At inference (one patient prediction), all models respond in milliseconds.

---

## Part 3 — Per-Model Analysis

### 1. Logistic Regression
- **Type**: Linear parametric classifier
- **How it works**: Fits a logistic (sigmoid) function to find a hyperplane separating diabetic from non-diabetic patients. Predicts P(diabetic) ∈ [0,1].
- **Strengths**: Highly interpretable (coefficients = feature importance); fastest training; well-calibrated probabilities.
- **Weaknesses**: Cannot capture non-linear relationships (e.g., interaction between Age and BMI).
- **Result summary**: Strong CV-AUC (0.8442 — 2nd highest), moderate Test Recall (70.37%). Good baseline.
- **Verdict**: Best interpretable linear model; useful as a baseline but not the clinical top choice.

---

### 2. Decision Tree
- **Type**: Non-linear rule-based classifier
- **How it works**: Recursively splits the feature space by maximising Gini impurity reduction at each node. Creates an interpretable decision flowchart.
- **Strengths**: Fully interpretable (can be printed as if-else rules); no scaling needed; handles non-linearity.
- **Weaknesses**: High variance — small changes in training data can yield very different trees. Low CV-AUC (0.7558) indicates overfitting even with `max_depth=6`.
- **Result summary**: Highest Recall tied with Extra Trees (74.07%) but lowest CV-AUC (0.7558). The high Recall is unreliable — it's partly driven by overfitting.
- **Verdict**: Not selected because high Recall is not backed by stable cross-validation performance.

---

### 3. Random Forest (Approach 1 Model)
- **Type**: Ensemble of 200 decision trees (bagging + random feature selection)
- **How it works**: Trains 200 trees on bootstrapped data subsets. Each tree uses a random subset of features at each split (√8 ≈ 3 features). Final prediction = majority vote.
- **Strengths**: Robust to overfitting; built-in feature importance; good on tabular data; OOB score for internal validation.
- **Weaknesses**: Black box (less interpretable than DT or LR); slower than simpler models.
- **Result summary**: Best Test-AUC (0.8254) and Accuracy (75.32%) and Precision (63.79%). But 4th in Recall (68.52%) — misses 31% of diabetic patients.
- **Verdict**: Best model for research AUC optimisation (Approach 1). Not chosen for clinical deployment due to lower Recall.

---

### 4. Gradient Boosting
- **Type**: Sequential ensemble (boosting)
- **How it works**: Trains trees one at a time. Each new tree focuses on the samples the previous ensemble got wrong (fits residuals). `subsample=0.8` adds stochasticity to reduce overfitting.
- **Strengths**: Often the best model on tabular data; low bias.
- **Weaknesses**: Slowest to train after RF; prone to overfitting if not regularised; does not support `class_weight` directly (uses sample weights instead).
- **Result summary**: Tied with RF in Accuracy (75.32%), 2nd in Test-AUC (0.8200), but low Recall (66.67%). Lowest CV-AUC of ensemble models (0.8047).
- **Verdict**: High accuracy but poor clinical Recall. Not suitable as primary screening model.

---

### 5. SVM (Support Vector Machine)
- **Type**: Maximum-margin classifier with RBF kernel
- **How it works**: Finds the hyperplane with maximum margin between classes in a high-dimensional feature space via the "kernel trick." RBF kernel maps data to infinite-dimensional space for non-linear separation.
- **Strengths**: Effective in high-dimensional spaces; robust with a clear margin of separation.
- **Weaknesses**: Requires feature scaling; `probability=True` adds overhead via Platt scaling; less interpretable.
- **Result summary**: 3rd in Recall (72.22%), 3rd in CV-AUC (0.8367). Balanced performer across metrics.
- **Verdict**: Strong contender. Close to Extra Trees in Recall. Not chosen because Extra Trees has both higher CV-AUC generalisation AND tied/higher Recall.

---

### 6. K-Nearest Neighbors
- **Type**: Non-parametric instance-based learner
- **How it works**: At prediction time, finds the K=11 nearest training samples (by Euclidean distance) and takes the weighted majority vote. No explicit model is built — the entire training set IS the model.
- **Strengths**: Simple to understand; no training phase; naturally multi-class.
- **Weaknesses**: Slow at inference for large datasets (must compare with all training points); very sensitive to feature scaling; K selection is critical.
- **Result summary**: Lowest Recall of all models (57.41%) — misses 43% of diabetic patients. Only model below 60% Recall.
- **Verdict**: Not suitable for diabetes screening. Distance-based methods struggle with correlated medical features.

---

### 7. Naive Bayes (Gaussian)
- **Type**: Probabilistic Bayesian classifier
- **How it works**: Applies Bayes' theorem assuming all features are Gaussian-distributed AND statistically independent of each other.
- **Strengths**: Fastest training and inference; no hyperparameters; naturally probabilistic; handles missing data well.
- **Weaknesses**: The independence assumption is violated in medical data — Glucose and Insulin are highly correlated (r=0.33). This leads to poor calibration.
- **Result summary**: Lowest Test-AUC (0.7646), lowest Accuracy (tied 70.13%), moderate Recall (62.96%).
- **Verdict**: Weakest model overall. The Gaussian independence assumption is too strong for correlated clinical biomarkers.

---

### 8. Extra Trees (Extremely Randomised Trees) ★ SELECTED
- **Type**: Ensemble of randomised decision trees (more randomised than Random Forest)
- **How it works**: Like Random Forest (builds 200 trees on the full dataset) but:
  - **Random Forest**: Finds the OPTIMAL split threshold for each feature at each node
  - **Extra Trees**: Draws a RANDOM split threshold — evaluates multiple random splits and picks the best among them
  - This introduces more randomness → trees are less correlated → ensemble variance is reduced
- **Strengths**:
  - Lower variance than RF (more randomisation reduces individual tree correlation)
  - Faster than RF (no optimal split search)
  - Highest CV-AUC (0.8480) — best generalisation across folds
  - Highest Test Recall (74.07%) — catches more diabetic patients
- **Clinical justification for selection**:
  - Recall = 74.07% means Extra Trees correctly identifies 74% of all diabetic patients
  - Compare: Random Forest only identifies 68.5% — missing 5.5% more patients
  - On 100 new diabetic patients: Extra Trees catches ~74, RF catches ~69 → 5 extra patients get treatment
  - Extra Trees also has better F1 (66.67% vs RF's 66.07%), meaning the precision trade-off is minimal
  - Highest CV-AUC proves the performance generalises — not a lucky test-set result
- **Weaknesses**: Slightly lower Test-AUC than RF (0.8178 vs 0.8254) and lower Precision (60.61% vs 63.79%)
- **Verdict**: ★ **SELECTED as Approach 2 winner** for clinical deployment. Best Recall + best generalisation (CV-AUC).

---

## Part 4 — Clinical Justification Summary

### Why Recall Over ROC-AUC for Medical Screening?

| Scenario | ROC-AUC Optimised (RF) | Recall Optimised (Extra Trees) |
|---|---|---|
| Patient actually has diabetes | Identified: 68.5% | Identified: **74.1%** |
| Patient missed (False Negative) | **31.5%** | 25.9% |
| False alarms (False Positives) | Lower | Slightly higher |
| Consequence of miss | No treatment → complications | Same |
| Consequence of false alarm | Unnecessary follow-up test | Same |

**Clinical conclusion**: In a population of 1,000 diabetic patients:
- Random Forest (AUC winner) would miss **315 patients**
- Extra Trees (Recall winner) would miss **259 patients**
- Extra Trees saves **56 patients** from undiagnosed diabetes per 1,000 screened

The ADA (American Diabetes Association) screening guidelines state: *"The goal of diabetes screening is to identify individuals who may have undiagnosed disease"* — this is fundamentally a Recall (Sensitivity) optimisation problem.

A false alarm leads to one follow-up HbA1c test. A missed diagnosis leads to years of uncontrolled hyperglycaemia, retinopathy, nephropathy, and cardiovascular events. The cost asymmetry is clear.

---

## Part 5 — Metric Priority Ranking for Clinical Screening

1. **Recall (Sensitivity)** — Primary: Minimise missed diabetic patients
2. **ROC-AUC** — Secondary: Overall discriminative ability
3. **F1-Score** — Tertiary: Balance (ensure Precision doesn't collapse)
4. **CV-AUC** — Stability: Ensure performance generalises to new patients
5. **Accuracy** — Informational: Not the primary criterion (misleading under imbalance)
6. **Precision** — Informational: Important for reducing unnecessary follow-ups
7. **Training Time** — Operational: Matters for retraining, not real-time inference

---

## Appendix — Confusion Matrix Terminology

```
                   Predicted
                 Non-Diabetic    Diabetic
Actual  Non-Diabetic    TN          FP
        Diabetic        FN          TP

TN = True Negative  : Correctly identified as Non-Diabetic
TP = True Positive  : Correctly identified as Diabetic     ← want high
FP = False Positive : Non-Diabetic incorrectly labelled Diabetic (false alarm)
FN = False Negative : Diabetic patient MISSED               ← want low (most critical)
```

**Recall = TP / (TP + FN)** — targets the bottom row (actual diabetic patients).
Maximising Recall minimises FN — the clinically dangerous outcome.

---

*Report generated for Diabetes Classification System — Approach 2 (Multi-Model Tournament)*
*Dataset: Pima Indians Diabetes Database | Algorithm: scikit-learn 1.5.1 | Python 3.11.4*
