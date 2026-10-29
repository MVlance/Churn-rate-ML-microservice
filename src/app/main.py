from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.serving.inference import predict as predict_model

app = FastAPI(
    title="Churn Prediction",
    description="ML decision for customers churn prediction",
    version="1.0.0",
)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def health_check():
    return {"status": "ok", "service": "predict churn rate"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        prediction = predict_model(data.model_dump())
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))