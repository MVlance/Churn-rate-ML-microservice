# 🔮 Telco Customer Churn Prediction — End-to-End MLOps Pipeline

An end-to-end Machine Learning & MLOps solution for predicting telecom customer churn. This project transitions an experimental machine learning workflow into a modular, production-ready microservice using **XGBoost**, **Optuna**, **MLflow**, **FastAPI**, **Docker**, and **AWS EC2** with a fully automated **GitHub Actions CI/CD** pipeline.

---

## 📌 Project Overview

Customer churn prediction allows telecom providers to proactively retain customers at risk of leaving. 

Instead of relying on the standard $0.50$ classification threshold, this system optimizes for a **business-driven decision threshold ($\tau = 0.15$)**, achieving **$\sim 90\%$ Recall** on churning customers while maintaining strong PR-AUC metrics ($\sim 0.66$).

---

## 🛠️ Tech Stack & Tools

* **Core Language:** Python 3.11
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy
* **Hyperparameter Optimization:** Optuna
* **Experiment Tracking & Artifact Management:** MLflow
* **API Framework & Validation:** FastAPI, Pydantic, Uvicorn
* **Containerization:** Docker, Docker Hub
* **Cloud Infrastructure:** AWS EC2 (Ubuntu Instance)
* **CI/CD Pipeline:** GitHub Actions

---

## 🏗️ Project Architecture

```text
ChurnProj/
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD pipeline for GitHub Actions
├── data/
│   └── raw/
│       └── churn_data.csv        # Telco Churn dataset
├── notebooks/
│   └── EDA.ipynb                 # Exploratory Data Analysis
├── src/
│   ├── app/
│   │   └── main.py               # FastAPI entry point & Pydantic schemas
│   ├── data/
│   │   ├── load_data.py          # Data loader module
│   │   └── preprocessor.py      # ColumnTransformer & feature engineering
│   ├── models/
│   │   ├── churn_pipeline.joblib # Serialized Scikit-Learn/XGBoost Pipeline
│   │   ├── evaluate.py           # Evaluation metrics helper
│   │   ├── train.py              # Training script integrated with MLflow
│   │   └── tune.py               # Optuna hyperparameter optimization
│   └── serving/
│       └── inference.py          # Model inference execution engine
├── Dockerfile                    # Container build configuration
├── .dockerignore
├── .gitignore
└── requirements.txt              # Project dependencies
```

---

## 🎯 ML Strategy and Decision Threshold

Standard classifiers default to a threshold of $\tau = 0.50$, which is sub-optimal for imbalanced retention problems where missing a churner (**False Negative**) is vastly more expensive than a false retention offer (**False Positive**).

* **Primary Optimization Metric:** PR-AUC (Precision-Recall Area Under Curve $\approx 0.66$)
* **Custom Business Threshold:** $\tau = 0.15$
* **Performance at $\tau = 0.15$:**
  * **Recall (Class 1):** $\approx 0.90$ ($90\%$ of churners detected)
  * **Precision (Class 1):** $\approx 0.46$

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.11+
* Docker Engine
* Git

### 2. Installation & Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/YOUR_USERNAME/ChurnProj.git
cd ChurnProj

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔬 Model Training & Experiment Tracking

To run hyperparameter tuning with **Optuna** and log metrics/artifacts with **MLflow**:

```bash
python -m src.models.train
```

To launch the MLflow UI and view logged experiments:

```bash
mlflow ui
```
Access the dashboard at `http://127.0.0.1:5000`.

---

## 🌐 Local API Serving (FastAPI)

To start the local Uvicorn development server:

```bash
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive OpenAPI / Swagger documentation will be available at:
* **Swagger UI:** `http://127.0.0.1:8000/docs`
* **Health Check:** `http://127.0.0.1:8000/`

---

## 🐳 Dockerization

### Build the Image
```bash
docker build -t churn-api:v1 .
```

### Run Container Locally
```bash
docker run -d -p 8000:8000 --name churn_container churn-api:v1
```

---

## ☁️ Deployment & CI/CD Pipeline

The project uses **GitHub Actions** for continuous integration and deployment to an **AWS EC2** instance:

1. **Build Step:** On every `git push` to `main`, GitHub Actions builds the Docker image.
2. **Registry Step:** Pushes the versioned image to **Docker Hub** (`YOUR_USERNAME/churn-api:latest`).
3. **Deploy Step:** Connects via SSH to the AWS EC2 instance, pulls the updated image, and restarts the live container.

---

## 📥 Sample API Payload & Response

### Request (`POST /predict`)
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "tenure": 2,
  "MonthlyCharges": 89.85,
  "TotalCharges": 179.70
}
```

### Response (`200 OK`)
```json
{
  "churn_probability": 0.8142,
  "will_churn": true,
  "risk_level": "High",
  "threshold_used": 0.15
}
```