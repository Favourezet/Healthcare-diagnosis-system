# Intelligent Healthcare Diagnosis and Decision Support System

You enter a patient's symptoms and vital signs. A machine learning model predicts the three most likely conditions, and a set of rules decides how urgent the case is and what should happen next (tests, which specialist, care advice). Every consultation is saved so you can look at it later.

It is a class project. Don't use it to diagnose real patients.

## How to run it

You need Python 3.10 or newer.

```
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The app opens in your browser at http://localhost:8501. If you skip `train_model.py`, the app trains the model by itself on its first run.

## Files

| File | What it does |
|---|---|
| `knowledge_base.py` | The 19 diseases and 60 symptoms, plus each disease's description, severity, tests, specialist and advice |
| `train_model.py` | Generates the training data, trains three models, compares them and saves the best one |
| `decision_support.py` | Runs the prediction and applies the triage rules |
| `app.py` | The web interface (Streamlit) |
| `model.pkl` | The trained model (created by `train_model.py`) |
| `model_report.txt` | Accuracy and per-disease results (created by `train_model.py`) |
| `data/training_data.csv` | The generated training data |
| `patient_records.csv` | Saved consultations (created when you run a diagnosis) |

## How it works

**1. Knowledge base.** Each disease has a list of symptoms and how often each one appears in people who have it. For example, fever appears in about 95% of malaria cases and chills in about 85%.

**2. Training data.** `train_model.py` uses those frequencies to create 300 fake patients per disease (5,700 in total). Each fake patient reports only 50 to 100% of their symptoms, because real patients don't mention everything. A few unrelated symptoms are added at random as noise.

**3. Machine learning.** Three models are trained on 80% of the data and tested on the other 20%:

| Model | Test accuracy |
|---|---|
| Naive Bayes | 83.0% |
| Random Forest | 78.8% |
| Decision Tree | 66.8% |

Naive Bayes did best, so the app uses it. The accuracy isn't higher because many diseases share symptoms (malaria, typhoid and Lassa fever all start with fever and headache), and some of the test patients report only two or three symptoms. That's why the app shows the top three conditions instead of only one.

**4. Decision support.** After the prediction, rules set one of four triage levels:

- **Self-care**: manage at home
- **Routine**: clinic visit in the next few days
- **Urgent**: see a doctor within 24 hours
- **Emergency**: go to hospital now

The starting level comes from how severe the predicted disease is. The rules then raise it when they find:

- danger signs (coughing blood, unusual bleeding, confusion, signs of dehydration)
- chest pain together with shortness of breath
- temperature of 40°C or above, or below 35°C
- blood pressure of 180/120 or above, or very low blood pressure
- high fever (39°C or more)
- fever in a child under 5 or an adult aged 65 or over

The app also warns you when the model is less than 40% sure, when the top two diseases are almost tied, and when fewer than three symptoms were entered.

## Diseases covered

Malaria, Typhoid Fever, Lassa Fever, Cholera, Tuberculosis, Pneumonia, COVID-19, Influenza, Common Cold, Gastroenteritis, Urinary Tract Infection, Type 2 Diabetes, Hypertension, Asthma, Migraine, Peptic Ulcer, Chickenpox, Anaemia, Viral Hepatitis.

## Limitations

- The training data is generated from estimated symptom frequencies, not real hospital records. Swapping in a real dataset would make the results more trustworthy.
- It only knows 19 diseases. Anything else will be matched to whichever of the 19 looks closest.
- The triage rules are simplified and haven't been reviewed by a clinician.

## Adding a disease

Add an entry to `DISEASES` in `knowledge_base.py` (and any new symptoms to `SYMPTOMS`), then run `python train_model.py` again.
