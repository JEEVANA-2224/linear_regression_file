import streamlit as st
import pickle
import os
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction App")
st.write("Predict **10-Year Coronary Heart Disease risk**")

# ---------------- Load Model & Scaler ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, "logistic_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model()

# ---------------- User Inputs ----------------
age = st.number_input("Age", 18, 100)
sys_bp = st.number_input("Systolic BP", 80, 200)
dia_bp = st.number_input("Diastolic BP", 50, 150)
chol = st.number_input("Cholesterol", 100, 400)
glucose = st.number_input("Glucose", 50, 300)
bmi = st.number_input("BMI", 10.0, 50.0)
heart_rate = st.number_input("Heart Rate", 40, 200)
cigs = st.number_input("Cigarettes per Day", 0, 100)

# ---------------- Prediction ----------------
if st.button("Predict"):
    input_data = np.array([[age, sys_bp, dia_bp, chol, glucose, bmi, heart_rate, cigs]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
