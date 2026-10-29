import joblib
import pandas as pd
from src.data.preprocessor import transform_raw_data

MODEL_PATH = "src/models/churn_pipeline.joblib"
pipeline = joblib.load(MODEL_PATH)
CUSTOM_THRESHOLD = 0.15

def predict(customer_data: dict) -> dict:
    df = pd.DataFrame([customer_data])

    transformed_data = transform_raw_data(df)

    probability = float(pipeline.predict_proba(transformed_data)[:, 1][0])

    will_churn = bool(probability >= CUSTOM_THRESHOLD)

    return {
        "churn_probability": probability,
        "will_churn": will_churn,
        "risk_level": "High" if will_churn else "Low",
        "threshold_used": CUSTOM_THRESHOLD
    }