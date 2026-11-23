"""
app.py - the web interface.

Run it with:  streamlit run app.py
"""

import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from decision_support import diagnose
from knowledge_base import DISEASES, SYMPTOMS

MODEL_FILE = "model.pkl"
RECORDS_FILE = "patient_records.csv"

st.set_page_config(page_title="Healthcare Diagnosis & Decision Support", page_icon="🩺")


@st.cache_resource
def load_model():
    # train the model the first time the app runs
    if not os.path.exists(MODEL_FILE):
        from train_model import train
        train()
    return joblib.load(MODEL_FILE)


def percent(p):
    return ">99%" if p >= 0.995 else f"{p * 100:.0f}%"


def save_record(name, age, sex, temperature, systolic, diastolic, result):
    bp = f"{systolic}/{diastolic}" if systolic and diastolic else ""
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": name,
        "age": age,
        "sex": sex,
        "temperature": temperature if temperature is not None else "",
        "blood_pressure": bp,
        "symptoms": "; ".join(SYMPTOMS[s] for s in result["symptoms"]),
        "diagnosis": result["top_disease"],
        "confidence": percent(result["confidence"]),
        "triage": result["triage"],
    }
    df = pd.DataFrame([row])
    df.to_csv(RECORDS_FILE, mode="a", index=False, header=not os.path.exists(RECORDS_FILE))


bundle = load_model()

st.title("🩺 Healthcare Diagnosis & Decision Support")
st.caption("For learning and demonstration only. This system does not replace a doctor.")

tab_diagnose, tab_records, tab_about = st.tabs(["New diagnosis", "Patient records", "About"])

# ---------------- Diagnosis tab ----------------
with tab_diagnose:
    with st.form("patient_form"):
        st.subheader("Patient details")
        name = st.text_input("Full name")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=0, max_value=120, value=25)
        sex = c2.selectbox("Sex", ["Female", "Male"])

        st.subheader("Symptoms")
        options = sorted(SYMPTOMS, key=lambda s: SYMPTOMS[s])
        chosen = st.multiselect("Select all the symptoms the patient has", options,
                                format_func=lambda s: SYMPTOMS[s])

        with st.expander("Vital signs (optional)"):
            temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0,
                                          value=None, step=0.1, placeholder="e.g. 37.0")
            c3, c4 = st.columns(2)
            systolic = c3.number_input("Systolic BP (top number)", min_value=50, max_value=260,
                                       value=None, placeholder="e.g. 120")
            diastolic = c4.number_input("Diastolic BP (bottom number)", min_value=30, max_value=160,
                                        value=None, placeholder="e.g. 80")

        submitted = st.form_submit_button("Diagnose", type="primary", width="stretch")

    if submitted:
        if not name.strip():
            st.warning("Please enter the patient's name.")
        elif not chosen and temperature is None:
            st.warning("Please select at least one symptom.")
        else:
            result = diagnose(bundle, chosen, age=age, temperature=temperature,
                              systolic=systolic, diastolic=diastolic)
            save_record(name.strip(), age, sex, temperature, systolic, diastolic, result)
            info = result["info"]

            st.divider()
            st.subheader("Result")

            # triage box, coloured by urgency
            box = {"Emergency": st.error, "Urgent": st.warning,
                   "Routine": st.info, "Self-care": st.success}[result["triage"]]
            box(f"**Triage level: {result['triage']}**\n\n{result['triage_meaning']}")

            st.markdown(f"### Most likely: {result['top_disease']}")
            st.write(info["description"])

            st.markdown("**Possible conditions**")
            for disease, prob in result["predictions"]:
                st.progress(float(prob), text=f"{disease}: {percent(prob)}")

            for w in result["warnings"]:
                st.warning(w)

            st.markdown("**Why this triage level**")
            for r in result["reasons"]:
                st.markdown(f"- {r}")

            c5, c6 = st.columns(2)
            with c5:
                st.markdown("**Recommended tests**")
                for t in info["tests"]:
                    st.markdown(f"- {t}")
            with c6:
                st.markdown("**Who to see**")
                st.write(info["specialist"])

            st.markdown("**Advice**")
            for a in info["advice"]:
                st.markdown(f"- {a}")

            st.caption("Saved to patient records.")

# ---------------- Records tab ----------------
with tab_records:
    st.subheader("Past consultations")
    if os.path.exists(RECORDS_FILE):
        records = pd.read_csv(RECORDS_FILE)
        st.dataframe(records.iloc[::-1], width="stretch", hide_index=True)
        st.download_button("Download records (CSV)", records.to_csv(index=False),
                           file_name="patient_records.csv", mime="text/csv")
    else:
        st.info("No consultations yet.")

# ---------------- About tab ----------------
with tab_about:
    st.subheader("How it works")
    st.markdown(f"""
1. **Input** – the user enters patient details, symptoms and (optionally) vital signs.
2. **Machine learning** – a **{bundle['model_name']}** model trained on symptom data predicts
   the three most likely conditions out of {len(DISEASES)}.
3. **Decision support** – rules check for danger signs, abnormal vital signs and high-risk ages,
   then set a triage level: Self-care, Routine, Urgent or Emergency.
4. **Recommendations** – the system shows tests to run, which specialist to see and care advice.
5. **Records** – every consultation is saved so it can be reviewed later.
""")
    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame({
        "Model": list(bundle["all_results"]),
        "Test accuracy": [f"{a * 100:.1f}%" for a in bundle["all_results"].values()],
    }), hide_index=True)
    st.subheader("Conditions covered")
    st.write(", ".join(sorted(DISEASES)))
