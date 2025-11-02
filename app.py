import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# Load model and scaler
@st.cache_resource
def load_artifacts():
    model = xgb.Booster()
    model.load_model("xgb_credit_fraud.model")
    scaler = joblib.load("scaler_amount.pkl")
    return model, scaler

model, scaler = load_artifacts()

# Page title
st.title("Credit Card Fraud Detection App")
st.write("Predict whether a transaction is fraudulent using a trained XGBoost model.")

# Sidebar for user input
st.sidebar.header("Enter Transaction Details")

# Input fields for all features
input_data = {}
for i in range(1, 29):
    input_data[f"V{i}"] = st.sidebar.number_input(f"V{i}", value=0.0, step=0.01)

amount = st.sidebar.number_input("Amount", value=50.0, step=1.0)
input_data["Amount"] = amount

# Create a dataframe
user_df = pd.DataFrame([input_data])

# Apply scaling
user_df["Amount"] = scaler.transform(user_df[["Amount"]])

# Make prediction
dmatrix = xgb.DMatrix(user_df)
prob = model.predict(dmatrix)[0]
label = int(prob >= 0.5)

# Display results
st.subheader("Prediction Result")
st.write(f"**Fraud Probability:** {prob:.4f}")
if label == 1:
    st.error("This transaction is predicted as FRAUDULENT.")
else:
    st.success("This transaction is predicted as LEGITIMATE.")
