# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load trained artifacts
# -----------------------------
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

# MUST match model_training.py
FEATURES = [
    "age",
    "income",
    "employment_years",
    "loan_amount",
    "loan_tenure",
    "emi",
    "credit_utilization",
    "num_loans",
    "past_due_30",
    "past_due_60",
    "past_due_90"
]

# -----------------------------
# UI
# -----------------------------
st.title("📊 Credit Risk Scorecard")
st.write("Estimate Probability of Default (PD) for a borrower")

# -----------------------------
# Inputs
# -----------------------------
age = st.number_input("Age", 18, 75, 30)
income = st.number_input("Annual Income", 10_000, 10_000_000, 500_000)
employment_years = st.number_input("Employment Years", 0, 40, 5)
loan_amount = st.number_input("Loan Amount", 10_000, 5_000_000, 300_000)
loan_tenure = st.number_input("Loan Tenure (months)", 6, 360, 36)
emi = st.number_input("Monthly EMI", 1_000, 200_000, 10_000)
credit_utilization = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.3)
num_loans = st.number_input("Number of Active Loans", 0, 10, 1)
past_due_30 = st.number_input("30+ Days Past Due", 0, 10, 0)
past_due_60 = st.number_input("60+ Days Past Due", 0, 10, 0)
past_due_90 = st.number_input("90+ Days Past Due", 0, 10, 0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Probability of Default"):
    input_data = pd.DataFrame([{
        "age": age,
        "income": income,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_tenure": loan_tenure,
        "emi": emi,
        "credit_utilization": credit_utilization,
        "num_loans": num_loans,
        "past_due_30": past_due_30,
        "past_due_60": past_due_60,
        "past_due_90": past_due_90
    }])

    # Enforce correct feature order
    input_data = input_data[FEATURES]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict PD
    pd_score = model.predict_proba(input_scaled)[0][1]

    # Risk Banding
    if pd_score < 0.2:
        risk = "Low Risk"
    elif pd_score < 0.5:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    # Display
    st.subheader("📌 Prediction Result")
    st.metric("Probability of Default", f"{pd_score:.2%}")
    st.metric("Risk Category", risk)