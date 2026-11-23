"""
decision_support.py

Takes the patient's symptoms and vital signs, asks the ML model for the most
likely diseases, then applies simple medical rules to decide how urgent the
case is and what the patient should do next.
"""

import pandas as pd

from knowledge_base import DISEASES, SYMPTOMS, expand_symptoms

# Triage levels, from least to most urgent
LEVELS = ["Self-care", "Routine", "Urgent", "Emergency"]
LEVEL_MEANING = {
    "Self-care": "Can be managed at home. See a doctor if it gets worse or lasts more than 3 days.",
    "Routine": "Book a clinic visit within the next few days.",
    "Urgent": "See a doctor within 24 hours.",
    "Emergency": "Go to the nearest hospital emergency unit now.",
}
SEVERITY_LEVEL = {"mild": 0, "moderate": 1, "severe": 2}

# Danger signs that always mean emergency
DANGER_SIGNS = {"coughing_blood", "unusual_bleeding", "confusion", "sunken_eyes"}

LOW_CONFIDENCE = 0.40   # below this, the system says it isn't sure


def vitals_to_symptoms(temperature):
    """Turn a temperature reading into symptoms the model understands."""
    found = set()
    if temperature is not None:
        if temperature >= 38.0:
            found.add("fever")
        if temperature >= 39.0:
            found.add("high_fever")
    return found


def predict(bundle, symptoms, top_n=3):
    """Return the top N (disease, probability) pairs from the ML model."""
    columns = bundle["symptoms"]
    x = pd.DataFrame([[1 if s in symptoms else 0 for s in columns]], columns=columns)
    probs = bundle["model"].predict_proba(x)[0]
    ranked = sorted(zip(bundle["model"].classes_, probs), key=lambda p: p[1], reverse=True)
    return ranked[:top_n]


def diagnose(bundle, symptoms, age=None, temperature=None, systolic=None, diastolic=None):
    """
    Main function the app calls.
    Returns a dictionary with predictions, triage level, reasons and advice.
    """
    symptoms = expand_symptoms(set(symptoms) | vitals_to_symptoms(temperature))
    predictions = predict(bundle, symptoms)
    top_disease, top_prob = predictions[0]
    info = DISEASES[top_disease]

    level = SEVERITY_LEVEL[info["severity"]]
    reasons = [f"Most likely condition ({top_disease}) is rated {info['severity']}."]
    warnings = []

    # --- rules that raise the urgency ---
    danger = symptoms & DANGER_SIGNS
    if danger:
        level = 3
        names = ", ".join(SYMPTOMS[s].lower() for s in sorted(danger))
        reasons.append(f"Danger sign present: {names}.")

    if {"chest_pain", "shortness_of_breath"} <= symptoms:
        level = 3
        reasons.append("Chest pain together with shortness of breath.")

    if temperature is not None:
        if temperature >= 40.0:
            level = 3
            reasons.append(f"Very high temperature ({temperature:.1f}°C).")
        elif temperature < 35.0:
            level = 3
            reasons.append(f"Body temperature is too low ({temperature:.1f}°C).")

    if systolic and diastolic:
        if systolic >= 180 or diastolic >= 120:
            level = 3
            reasons.append(f"Blood pressure is dangerously high ({systolic}/{diastolic} mmHg).")
        elif systolic >= 140 or diastolic >= 90:
            level = max(level, 1)
            warnings.append(f"Blood pressure reading is high ({systolic}/{diastolic} mmHg). "
                            "Get it rechecked and see a doctor about hypertension.")
        elif systolic < 90 or diastolic < 60:
            level = max(level, 2)
            reasons.append(f"Blood pressure is low ({systolic}/{diastolic} mmHg).")

    if "high_fever" in symptoms and level < 2:
        level = 2
        reasons.append("High fever (39°C or more) needs a doctor's review soon.")

    if age is not None and "fever" in symptoms and (age < 5 or age >= 65):
        level = max(level, 2)
        reasons.append("Fever in a child under 5 or an adult aged 65 and above.")

    # --- rules about how much to trust the prediction ---
    if top_prob < LOW_CONFIDENCE:
        level = max(level, 1)
        warnings.append("The system is not confident about this result. "
                        "A doctor needs to examine the patient.")

    second_disease, second_prob = predictions[1]
    if top_prob - second_prob < 0.15 and second_prob > 0.2:
        warnings.append(f"{top_disease} and {second_disease} are almost equally likely. "
                        "A lab test is needed to tell them apart.")

    if len(symptoms) < 3:
        warnings.append("Only a few symptoms were entered. Add more for a better result.")

    triage = LEVELS[level]
    return {
        "symptoms": sorted(symptoms),
        "predictions": predictions,
        "top_disease": top_disease,
        "confidence": top_prob,
        "info": info,
        "triage": triage,
        "triage_meaning": LEVEL_MEANING[triage],
        "reasons": reasons,
        "warnings": warnings,
    }
