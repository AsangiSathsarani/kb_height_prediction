import streamlit as st
import joblib
import pandas as pd

# Load model & scaler
model = joblib.load("KB_missing_hight_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("KB Height Bulk Prediction (Upload File)")

# Upload file
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "csv"])

if uploaded_file is not None:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Check required columns
    required_cols = ["FIRST_ELEVATION", "GROUND_ELEVATION", "KB HEIGHT"]

    if not all(col in df.columns for col in required_cols):
        st.error(f"File must contain columns: {required_cols}")
    else:

        # Find missing KB HEIGHT rows
        missing_mask = df["KB HEIGHT"].isna()

        if missing_mask.sum() == 0:
            st.warning("No missing KB HEIGHT values found.")
        else:
            st.write(f"Missing KB HEIGHT rows: {missing_mask.sum()}")

            # Prepare input for prediction
            input_df = df.loc[missing_mask, ["FIRST_ELEVATION", "GROUND_ELEVATION"]]

            # Scale
            scaled_input = scaler.transform(input_df)

            # Predict
            predictions = model.predict(scaled_input)

            # Fill missing values
            df.loc[missing_mask, "KB HEIGHT"] = predictions

            st.success("Missing KB HEIGHT values filled successfully!")

            st.dataframe(df.head())

            # Download updated file
            output_file = "updated_kb_height.xlsx"
            df.to_excel(output_file, index=False)

            with open(output_file, "rb") as f:
                st.download_button(
                    label="Download Updated File",
                    data=f,
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )