import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn")

model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0, 200, 50)
total_charges = st.number_input("Total Charges", 0, 10000, 1000)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer", "Credit card"
])

input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract],
    "InternetService": [internet_service],
    "PaymentMethod": [payment_method]
})

input_encoded = pd.get_dummies(input_data)

# Align columns with training data
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

input_scaled = scaler.transform(input_encoded)

if st.button("Predict"):

    prediction_proba = model.predict_proba(input_scaled)[0][1]

    if prediction_proba > 0.5:
        st.error(f"High chance of churn ({prediction_proba*100:.1f}%)")
    else:
        st.success(f"Low chance of churn ({prediction_proba*100:.1f}%)")
