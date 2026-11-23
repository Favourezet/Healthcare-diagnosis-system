"""
train_model.py

1. Generates a training dataset of fake patients from the knowledge base.
2. Trains three machine learning models and compares them.
3. Saves the best model to model.pkl so the app can use it.

Run it with:  python train_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from knowledge_base import DISEASES, SYMPTOMS, IMPLIES

SAMPLES_PER_DISEASE = 300   # fake patients per disease
NOISE = 0.02                # chance an unrelated symptom shows up anyway
MODEL_FILE = "model.pkl"
DATA_FILE = os.path.join("data", "training_data.csv")
REPORT_FILE = "model_report.txt"


def generate_dataset(seed=42):
    """Create one row per fake patient: 1 = has the symptom, 0 = doesn't."""
    rng = np.random.default_rng(seed)
    symptom_list = list(SYMPTOMS)
    rows, labels = [], []

    for disease, info in DISEASES.items():
        made = 0
        while made < SAMPLES_PER_DISEASE:
            # Real patients don't mention every symptom they have.
            # Each fake patient reports between 50% and 100% of their symptoms.
            report_rate = rng.uniform(0.5, 1.0)
            patient = {}
            for s in symptom_list:
                if s in info["symptoms"]:
                    p = info["symptoms"][s] * report_rate
                else:
                    p = NOISE
                patient[s] = int(rng.random() < p)

            # specific symptom means the general one is there too (dry cough -> cough)
            for specific, general in IMPLIES.items():
                if patient[specific]:
                    patient[general] = 1

            # skip patients with fewer than 2 symptoms, they tell us nothing useful
            if sum(patient.values()) < 2:
                continue

            rows.append([patient[s] for s in symptom_list])
            labels.append(disease)
            made += 1

    df = pd.DataFrame(rows, columns=symptom_list)
    df["disease"] = labels
    return df


def train():
    df = generate_dataset()
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_FILE, index=False)

    X = df.drop(columns="disease")
    y = df["disease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Naive Bayes": BernoulliNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = accuracy_score(y_test, model.predict(X_test))

    best_name = max(results, key=results.get)
    best_model = models[best_name]

    # write a short report you can paste into your project write-up
    lines = [f"Dataset: {len(df)} patients, {len(SYMPTOMS)} symptoms, {len(DISEASES)} diseases",
             f"Train/test split: 80% / 20%", "", "Model accuracy on the test set:"]
    for name, acc in results.items():
        lines.append(f"  {name:<14} {acc * 100:.2f}%")
    lines += ["", f"Best model: {best_name}", "",
              classification_report(y_test, best_model.predict(X_test))]
    report = "\n".join(lines)
    with open(REPORT_FILE, "w") as f:
        f.write(report)
    print(report)

    joblib.dump({
        "model": best_model,
        "symptoms": list(X.columns),
        "model_name": best_name,
        "accuracy": results[best_name],
        "all_results": results,
    }, MODEL_FILE)
    print(f"Saved {best_name} to {MODEL_FILE}")


if __name__ == "__main__":
    train()
