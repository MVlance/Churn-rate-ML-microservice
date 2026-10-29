from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.preprocessor import transform_raw_data
from src.models.tune import tune_model
from src.data.preprocessor import get_preprocessor
from src.data.load_data import load_data
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, confusion_matrix, classification_report, average_precision_score
import mlflow
import joblib

def train_model(data_path: str, custom_threshold:float = 0.15):
    raw_df = load_data(data_path)

    df = transform_raw_data(raw_df)

    X = df.drop(columns='Churn')
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = get_preprocessor()
    X_train_preprocessor = preprocessor.fit_transform(X_train)

    print("Optuna tuning model...")
    best_params = tune_model(X_train_preprocessor, y_train)

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(**best_params))
    ])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        probas = pipeline.predict_proba(X_test)[:, 1]
        predicts = (probas >= custom_threshold).astype(int)

        rec = recall_score(y_test, predicts)
        pr_auc = average_precision_score(y_test, probas)
        conf_matrix = confusion_matrix(y_test, predicts)

        mlflow.log_artifact(data_path)
        mlflow.log_params(best_params)
        mlflow.log_param("custom_threshold", custom_threshold)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("pr-auc", pr_auc)
        mlflow.xgboost.log_model(pipeline.named_steps['clf'], name="model")

        joblib.dump(pipeline, "../models/churn_pipeline.joblib")
        print(f"Training completed. Final stats:\n Recall: {rec}, \n PR-AUC: {pr_auc}, \n Confusion Matrix: \n{conf_matrix}, \n Classification Report: \n{classification_report(y_test, predicts)}")

if __name__ == "__main__":
    train_model("/home/vladyslav/PycharmProjects/ChurnProj/src/data/raw/churn_data.csv")