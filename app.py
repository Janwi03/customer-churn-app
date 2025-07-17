import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd

# Set page config at the very top
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load model
MODEL_PATH = os.path.join("models", "churn_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.markdown(
    """
    <style>
    .main {
        background-color: #1e1e2f;
        color: white;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h3, p {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill the form below to predict if a customer is likely to churn.</p>", unsafe_allow_html=True)

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure_months = st.number_input("Tenure Months", min_value=0)
        churn_score = st.slider("Churn Score", 0, 100)
        gender_male = st.selectbox("Gender", ["Female", "Male"]) == "Male"
        partner_yes = st.selectbox("Has Partner", ["No", "Yes"]) == "Yes"
        phone_service_yes = st.selectbox("Phone Service", ["No", "Yes"]) == "Yes"
        internet_fiber = st.selectbox("Internet: Fiber Optic", ["No", "Yes"]) == "Yes"

    with col2:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
        cltv = st.number_input("CLTV", min_value=0.0)
        senior_citizen_yes = st.selectbox("Senior Citizen", ["No", "Yes"]) == "Yes"
        dependents_yes = st.selectbox("Has Dependents", ["No", "Yes"]) == "Yes"
        internet_none = st.selectbox("Internet: None", ["No", "Yes"]) == "Yes"
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    contract_one_year = contract_type == "One year"
    contract_two_year = contract_type == "Two year"

    submitted = st.form_submit_button("Predict Churn")

    if submitted:
        input_features = np.array([[
            tenure_months,
            monthly_charges,
            churn_score,
            cltv,
            float(gender_male),
            float(senior_citizen_yes),
            float(partner_yes),
            float(dependents_yes),
            float(phone_service_yes),
            float(internet_fiber),
            float(internet_none),
            float(contract_one_year),
            float(contract_two_year)
        ]])

        prediction = model.predict(input_features)[0]
        result_text = "Churn Likely" if prediction == 1 else "No Churn Expected"
        result_color = "#ff4d4d" if prediction == 1 else "#1faa59"

        st.markdown(f"""
            <h3 style='text-align: center; color: {result_color};'>
                Prediction: {result_text}
            </h3>
        """, unsafe_allow_html=True)