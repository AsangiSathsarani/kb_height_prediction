from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
import numpy as np
from io import BytesIO
from fastapi.responses import StreamingResponse

app = FastAPI(title="KB Height Prediction API")

# Load model & scaler
model = joblib.load("KB_missing_hight_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# 1. Single Prediction API
# -----------------------------
@app.post("/predict")
def predict(data: dict):

    try:
        input_df = pd.DataFrame([{
            "FIRST_ELEVATION": data["FIRST_ELEVATION"],
            "GROUND_ELEVATION": data["GROUND_ELEVATION"]
        }])

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)

        return {"predicted_kb_height": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# 2. Bulk Prediction API
# -----------------------------
@app.post("/bulk_predict")
async def bulk_predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        # Read file
        if file.filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(contents))
        else:
            df = pd.read_excel(BytesIO(contents))

        # Check required columns
        required_cols = ["FIRST_ELEVATION", "GROUND_ELEVATION", "KB HEIGHT"]

        if not all(col in df.columns for col in required_cols):
            return {"error": f"Missing required columns: {required_cols}"}

        # Find missing values
        missing_mask = df["KB HEIGHT"].isna()

        if missing_mask.sum() == 0:
            return {"message": "No missing KB HEIGHT values found."}

        # Predict
        input_df = df.loc[missing_mask, ["FIRST_ELEVATION", "GROUND_ELEVATION"]]
        scaled_input = scaler.transform(input_df)
        predictions = model.predict(scaled_input)

        # Fill values
        df.loc[missing_mask, "KB HEIGHT"] = predictions

        # Convert to Excel in memory
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=updated_file.xlsx"}
        )
    except Exception as e:
        return {"error": str(e)}