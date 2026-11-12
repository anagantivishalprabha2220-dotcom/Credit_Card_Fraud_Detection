
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model and scaler
with open('fraud_model_rf.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to check whether it is fraudulent.")

# Input fields
time = st.number_input("Transaction Time (seconds)", min_value=0.0, value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

st.subheader("PCA Features")

v_features = []

for i in range(1, 29):
    value = st.number_input(
        f"V{i}",
        value=0.0,
        key=f"v{i}"
    )
    v_features.append(value)

if st.button("Check Transaction"):

    # Feature engineering
    hour = (time // 3600) % 24
    log_amount = np.log1p(amount)
    high_amount_flag = 1 if amount > 200 else 0

    # Prepare input in correct order
    data = [time] + v_features + [
        amount, hour, log_amount, high_amount_flag
    ]

    columns = (
        ['Time'] +
        [f'V{i}' for i in range(1, 29)] +
        ['Amount', 'Hour', 'Log_Amount', 'High_Amount_Flag']
    )

    input_df = pd.DataFrame([data], columns=columns)

    # Scale the same columns used during training
    input_df[['Time', 'Amount', 'Log_Amount']] = scaler.transform(
        input_df[['Time', 'Amount', 'Log_Amount']]
    )

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Normal Transaction")

    st.write(f"Fraud Probability: {probability * 100:.2f}%")

st.warning(
    "Educational demonstration only. "
    "Zero-valued PCA features are placeholders, not real transaction data."
)

