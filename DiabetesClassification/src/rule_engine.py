"""
rule_engine.py
==============
Clinical Rule-Based Diabetes Type Classification Engine.

PURPOSE:
  This module is executed ONLY when the ML model predicts DIABETIC.
  It determines WHICH TYPE of diabetes based on documented clinical criteria.

WHY RULE-BASED (NOT ML)?
  The Pima Indians dataset contains only binary labels: 0 (no diabetes)
  or 1 (diabetes). There are NO type labels.
  Since you cannot train an ML model without labeled training data,
  we use evidence-based clinical rules instead.

  This approach is:
  ✓ Medically defensible (rules sourced from ADA guidelines)
  ✓ Explainable (you can trace exactly WHY a type was assigned)
  ✓ Academically acceptable for a dataset with no type labels
  ✓ Matches how real clinical decision support systems work

CLINICAL REFERENCES:
  - American Diabetes Association. Standards of Medical Care in Diabetes — 2023.
    Diabetes Care 2023;46(Suppl. 1):S1–S4.
  - Classification: Gestational, Type 1, Type 2, Prediabetes.
  - Diagnostic thresholds per ADA 2023 guidelines.

RULE LOGIC (IN PRIORITY ORDER):

  PREDIABETES CHECK (highest priority):
    Glucose 100–125 mg/dL (Impaired Fasting Glucose per ADA).
    Must check this first — prediabetes exists on the spectrum BEFORE Type 2.

  GESTATIONAL DIABETES:
    Pregnancy + elevated glucose + age in reproductive window.
    Clinically: GDM is diagnosed during pregnancy, glucose > 140 mg/dL
    on the 2-hour oral glucose tolerance test.

  TYPE 1 DIABETES:
    Typically: younger age (<30), lower BMI (<30), very low insulin.
    Pathophysiology: autoimmune destruction of beta cells → insulin deficiency.

  TYPE 2 DIABETES (default):
    Typically: older age, higher BMI, insulin resistance (high insulin
    but body doesn't respond to it properly).
    Accounts for 90–95% of all diabetes cases globally.

IMPORTANT: These rules are clinical heuristics, not absolute diagnoses.
  Real diagnosis requires a physician and additional tests (HbA1c, C-peptide, etc.)
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Data classes for structured results ───────────────────────────

@dataclass
class ClassificationResult:
    """Result of the rule-based classification."""

    diabetes_type: str                # e.g., "Type 2 Diabetes"
    confidence_label: str             # "High", "Moderate", "Low"
    rules_matched: list = field(default_factory=list)
    clinical_description: str = ""
    risk_message: str = ""
    recommendations: list = field(default_factory=list)
    icd_code: str = ""                # ICD-10 code for medical records


# ── Clinical thresholds (ADA 2023 Guidelines) ─────────────────────

class ClinicalThresholds:
    """
    Thresholds derived from ADA 2023 Standards of Medical Care.
    Centralizing them here makes the code auditable and easily updateable.
    """
    # Glucose thresholds (mg/dL)
    GLUCOSE_PREDIABETES_LOW  = 100
    GLUCOSE_PREDIABETES_HIGH = 125
    GLUCOSE_DIABETES         = 126
    GLUCOSE_GDM              = 140   # 2-hour OGTT threshold for GDM

    # BMI thresholds (kg/m²)
    BMI_NORMAL     = 25.0
    BMI_OVERWEIGHT = 25.0
    BMI_OBESE      = 30.0

    # Age thresholds (years)
    AGE_YOUNG         = 30           # Type 1 more common below this
    AGE_GDM_MIN       = 18
    AGE_GDM_MAX       = 45

    # Insulin thresholds (mu U/ml)
    INSULIN_LOW       = 80           # Low = beta cell dysfunction (Type 1 indicator)
    INSULIN_HIGH      = 150          # High = insulin resistance (Type 2 indicator)

    # Pregnancy
    MIN_PREGNANCIES_GDM = 1          # Must have been pregnant for GDM


T = ClinicalThresholds


# ── Core classification function ──────────────────────────────────

def classify_diabetes_type(
    glucose: float,
    bmi: float,
    age: float,
    insulin: float,
    pregnancies: float,
    blood_pressure: float = 72.0,
    diabetes_pedigree: float = 0.5,
) -> ClassificationResult:
    """
    Apply clinical rules to determine diabetes type.

    Parameters
    ----------
    glucose : float — Plasma glucose (mg/dL)
    bmi : float — Body Mass Index (kg/m²)
    age : float — Patient age (years)
    insulin : float — Serum insulin (mu U/ml)
    pregnancies : float — Number of pregnancies
    blood_pressure : float — Diastolic BP (mm Hg)
    diabetes_pedigree : float — Family history risk score

    Returns
    -------
    ClassificationResult with type, description, recommendations
    """

    rules_matched = []

    # ── RULE 1: PREDIABETES ────────────────────────────────────────
    # Priority: checked first because prediabetes exists at lower glucose
    # even in a patient classified as diabetic by the ML model.
    # (ML model may have a low threshold; clinical rule refines it.)
    if T.GLUCOSE_PREDIABETES_LOW <= glucose <= T.GLUCOSE_PREDIABETES_HIGH:
        rules_matched.append(
            f"Glucose {glucose:.0f} mg/dL is in prediabetes range "
            f"(100–125 mg/dL per ADA 2023)"
        )
        return ClassificationResult(
            diabetes_type="Prediabetes",
            confidence_label="High",
            rules_matched=rules_matched,
            clinical_description=(
                "Blood glucose is in the prediabetes range (100–125 mg/dL). "
                "This is NOT yet full diabetes but indicates higher-than-normal "
                "blood glucose. Without lifestyle changes, prediabetes often "
                "progresses to Type 2 diabetes within 5–10 years."
            ),
            risk_message=(
                "⚠ Your glucose is in the prediabetes range. "
                "Lifestyle intervention can prevent or delay Type 2 diabetes."
            ),
            recommendations=[
                "Lose 5–7% of body weight if overweight",
                "150 minutes per week of moderate physical activity",
                "Reduce refined carbohydrates and sugar intake",
                "Recheck HbA1c every 6–12 months",
                "Consider metformin if other risk factors are present (consult physician)",
            ],
            icd_code="R73.03",
        )

    # ── RULE 2: GESTATIONAL DIABETES ──────────────────────────────
    # ADA Criteria: diagnosed during pregnancy, typically with
    # 2-hour OGTT glucose ≥ 140 mg/dL.
    is_pregnant_age_range = T.AGE_GDM_MIN <= age <= T.AGE_GDM_MAX
    has_pregnancies = pregnancies >= T.MIN_PREGNANCIES_GDM
    high_glucose_for_gdm = glucose >= T.GLUCOSE_GDM

    if has_pregnancies and high_glucose_for_gdm and is_pregnant_age_range:
        rules_matched.append(
            f"Pregnancy history (n={pregnancies:.0f})"
        )
        rules_matched.append(
            f"Glucose {glucose:.0f} mg/dL ≥ {T.GLUCOSE_GDM} mg/dL "
            f"(GDM threshold)"
        )
        rules_matched.append(
            f"Age {age:.0f} years within gestational range "
            f"({T.AGE_GDM_MIN}–{T.AGE_GDM_MAX})"
        )
        return ClassificationResult(
            diabetes_type="Gestational Diabetes",
            confidence_label="Moderate",
            rules_matched=rules_matched,
            clinical_description=(
                "Gestational Diabetes Mellitus (GDM) develops during pregnancy "
                "due to hormonal changes that cause insulin resistance. "
                "It typically resolves after delivery but increases the risk of "
                "Type 2 diabetes later in life (40–60% chance within 10 years)."
            ),
            risk_message=(
                "⚠ Gestational diabetes detected based on pregnancy history and "
                "glucose levels. Regular monitoring is essential."
            ),
            recommendations=[
                "Continuous blood glucose monitoring during and after pregnancy",
                "Dietary management: low glycemic index foods",
                "Moderate exercise as approved by your obstetrician",
                "Insulin therapy if diet/exercise are insufficient",
                "Postpartum glucose testing at 4–12 weeks after delivery",
                "Annual diabetes screening thereafter",
            ],
            icd_code="O24.419",
        )

    # ── RULE 3: TYPE 1 DIABETES ────────────────────────────────────
    # Typical profile: young (<30), lean (BMI <30), low insulin
    # (beta cell destruction → body produces little/no insulin).
    # High glucose despite low insulin = insulin deficiency (Type 1)
    # vs insulin resistance (Type 2 where insulin is high but ineffective).
    is_young = age < T.AGE_YOUNG
    is_lean_or_normal = bmi < T.BMI_OBESE
    has_low_insulin = insulin < T.INSULIN_LOW
    high_glucose = glucose >= T.GLUCOSE_DIABETES

    type1_score = sum([is_young, is_lean_or_normal, has_low_insulin, high_glucose])

    if type1_score >= 3:
        if is_young:
            rules_matched.append(f"Age {age:.0f} < {T.AGE_YOUNG} (younger onset)")
        if is_lean_or_normal:
            rules_matched.append(f"BMI {bmi:.1f} < {T.BMI_OBESE} (lean/normal weight)")
        if has_low_insulin:
            rules_matched.append(
                f"Insulin {insulin:.0f} < {T.INSULIN_LOW} mu U/ml "
                f"(low — suggests beta cell dysfunction)"
            )
        if high_glucose:
            rules_matched.append(
                f"Glucose {glucose:.0f} ≥ {T.GLUCOSE_DIABETES} mg/dL "
                f"(elevated despite low insulin)"
            )

        confidence = "High" if type1_score == 4 else "Moderate"

        return ClassificationResult(
            diabetes_type="Type 1 Diabetes",
            confidence_label=confidence,
            rules_matched=rules_matched,
            clinical_description=(
                "Type 1 Diabetes is an autoimmune condition where the immune system "
                "destroys insulin-producing beta cells in the pancreas. "
                "The body produces little or no insulin, requiring lifelong "
                "insulin therapy. It is NOT caused by lifestyle factors and "
                "typically develops in younger individuals."
            ),
            risk_message=(
                "⚠ Profile consistent with Type 1 Diabetes. "
                "Immediate medical evaluation required for proper diagnosis "
                "(C-peptide test and autoantibody panel recommended)."
            ),
            recommendations=[
                "Immediate referral to an endocrinologist",
                "Insulin therapy (required — cannot be managed with diet alone)",
                "Continuous glucose monitoring (CGM)",
                "HbA1c test every 3 months",
                "Watch for DKA (diabetic ketoacidosis) symptoms",
                "Carbohydrate counting and insulin-to-carb ratio management",
            ],
            icd_code="E10.9",
        )

    # ── RULE 4: TYPE 2 DIABETES (default) ─────────────────────────
    # Most common form (90–95% of all diabetes).
    # Characterized by insulin resistance: pancreas still produces insulin
    # but cells don't respond to it properly.
    # Strongly associated with obesity, older age, sedentary lifestyle.
    rules_matched.append(
        f"Default classification — most prevalent type (90–95% of cases)"
    )

    if bmi >= T.BMI_OBESE:
        rules_matched.append(
            f"BMI {bmi:.1f} ≥ {T.BMI_OBESE} (obesity — major Type 2 risk factor)"
        )
    if insulin >= T.INSULIN_HIGH:
        rules_matched.append(
            f"Insulin {insulin:.0f} ≥ {T.INSULIN_HIGH} mu U/ml "
            f"(elevated — suggests insulin resistance)"
        )
    if age >= T.AGE_YOUNG:
        rules_matched.append(f"Age {age:.0f} ≥ {T.AGE_YOUNG} (adult onset)")
    if diabetes_pedigree > 0.5:
        rules_matched.append(
            f"Diabetes Pedigree {diabetes_pedigree:.3f} > 0.5 "
            f"(elevated family history risk)"
        )

    # Severity based on how far above normal
    if glucose >= 200:
        severity = "Severe — Immediate medical attention required"
    elif glucose >= 150:
        severity = "Moderate-to-High — Medical management needed"
    else:
        severity = "Moderate — Lifestyle changes + medical management"

    return ClassificationResult(
        diabetes_type="Type 2 Diabetes",
        confidence_label="High",
        rules_matched=rules_matched,
        clinical_description=(
            "Type 2 Diabetes is a metabolic disorder characterized by insulin "
            "resistance and progressive beta cell dysfunction. The body still "
            f"produces insulin but cells don't respond effectively. {severity}. "
            "It is strongly associated with obesity, physical inactivity, and "
            "family history. It is manageable with lifestyle changes, oral "
            "medications, and/or insulin."
        ),
        risk_message=(
            f"⚠ Profile consistent with Type 2 Diabetes ({severity}). "
            "Consult a physician immediately for proper diagnosis and treatment plan."
        ),
        recommendations=[
            "HbA1c test to confirm diagnosis and baseline (target < 7%)",
            "Lose 5–10% body weight if BMI > 30",
            "150 minutes/week of moderate aerobic exercise",
            "Low glycemic index diet, reduce refined carbohydrates",
            "Consider metformin (first-line oral medication) as prescribed",
            "Monitor fasting glucose daily",
            "Annual eye exam, kidney function tests, foot examination",
        ],
        icd_code="E11.9",
    )


# ── Non-Diabetic result ────────────────────────────────────────────

def get_non_diabetic_result(glucose: float, bmi: float, age: float) -> dict:
    """
    Return a structured result for non-diabetic prediction.
    Include risk awareness based on near-threshold values.
    """
    risk_level = "Low"
    risk_message = "Your current profile does not indicate diabetes."
    recommendations = [
        "Maintain healthy weight and regular physical activity",
        "Annual health check-ups recommended",
        "Balanced diet with limited refined sugars",
    ]

    # Warn if borderline
    if glucose >= 90:
        risk_level = "Borderline"
        risk_message = (
            f"Glucose {glucose:.0f} mg/dL is approaching the prediabetes range "
            f"(≥ 100 mg/dL). Monitor and maintain a healthy lifestyle."
        )
        recommendations.insert(0, "Recheck blood glucose in 6–12 months")

    if bmi >= 27:
        recommendations.append(
            "BMI is approaching overweight range — weight management recommended"
        )

    return {
        "prediction": "Non-Diabetic",
        "diabetes_type": None,
        "risk_level": risk_level,
        "risk_message": risk_message,
        "recommendations": recommendations,
        "icd_code": None,
    }


# ── Master function called by Flask ───────────────────────────────

def run_rule_engine(ml_prediction: int, input_features: dict) -> dict:
    """
    Entry point for the rule engine, called from predictor.py / app.py.

    Parameters
    ----------
    ml_prediction : int — 0 (Non-Diabetic) or 1 (Diabetic) from ML model
    input_features : dict — raw input values from the user form

    Returns
    -------
    dict with all result fields for the UI to display
    """
    glucose = float(input_features.get("Glucose", 0))
    bmi = float(input_features.get("BMI", 0))
    age = float(input_features.get("Age", 0))
    insulin = float(input_features.get("Insulin", 0))
    pregnancies = float(input_features.get("Pregnancies", 0))
    blood_pressure = float(input_features.get("BloodPressure", 72))
    diabetes_pedigree = float(input_features.get("DiabetesPedigreeFunction", 0.5))

    if ml_prediction == 0:
        return get_non_diabetic_result(glucose, bmi, age)

    # ML says diabetic → apply rule engine
    result = classify_diabetes_type(
        glucose=glucose,
        bmi=bmi,
        age=age,
        insulin=insulin,
        pregnancies=pregnancies,
        blood_pressure=blood_pressure,
        diabetes_pedigree=diabetes_pedigree,
    )

    return {
        "prediction": "Diabetic",
        "diabetes_type": result.diabetes_type,
        "confidence_label": result.confidence_label,
        "rules_matched": result.rules_matched,
        "clinical_description": result.clinical_description,
        "risk_message": result.risk_message,
        "recommendations": result.recommendations,
        "icd_code": result.icd_code,
    }
