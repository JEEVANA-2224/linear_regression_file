import streamlit as st
import numpy as np
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction App")
st.write("Predict **10-Year Coronary Heart Disease risk** using Logistic Regression.")

# ---------------- Load Model & Scaler ----------------
@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ---------------- User Inputs ----------------
st.subheader("🧾 Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0)
totChol = st.number_input("Total Cholesterol", min_value=100, max_value=600, value=200)
sysBP = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
diaBP = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)
BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
glucose = st.number_input("Glucose", min_value=40, max_value=300, value=90)
BPMeds = st.selectbox("On Blood Pressure Meds?", [0, 1])
prevalentStroke = st.selectbox("History of Stroke?", [0, 1])
prevalentHyp = st.selectbox("History of Hypertension?", [0, 1])
diabetes = st.selectbox("Diabetes?", [0, 1])
heartRate = st.number_input("Heart Rate", min_value=40, max_value=200, value=70)
education = st.selectbox("Education Level (0-4)", [0, 1, 2, 3, 4])

# ---------------- Prediction ----------------
if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cigsPerDay, totChol, sysBP, diaBP, BMI, glucose,
                            BPMeds, prevalentStroke, prevalentHyp, diabetes, heartRate, education]])
    
    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[:, 1][0]
    prediction = 1 if prob >= 0.3 else 0   # threshold tuned for recall

    st.subheader("📊 Result")
    st.write(f"**Risk Probability:** `{prob:.2f}`")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
