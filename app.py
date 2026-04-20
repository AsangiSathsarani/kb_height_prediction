import streamlit as st
import joblib
import pandas as pd

# Load
model = joblib.load("KB_missing_hight_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("KB Height Prediction")

# Inputs
first_elev = st.number_input("First Elevation")
ground_elev = st.number_input("Ground Elevation")

if st.button("Predict"):

    # ✅ Use DataFrame with column names
    input_df = pd.DataFrame([{
        "FIRST_ELEVATION": first_elev,
        "GROUND_ELEVATION": ground_elev
    }])

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)

    st.success(f"Predicted KB Height: {prediction[0]:.2f}")