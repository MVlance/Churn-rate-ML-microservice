import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def transform_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    dataframe = df.copy()

    dataframe.drop("customerID", axis=1, inplace=True, errors="ignore")

    if 'TotalCharges' in dataframe.columns:
        dataframe['TotalCharges'] = pd.to_numeric(dataframe['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)

    binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    replacement = {
        "Yes": 1, "No": 0,
        "Male": 1, "Female": 0
    }

    for column in dataframe.columns:
        if column in binary_columns:
            dataframe[column] = dataframe[column].map(replacement).astype(int)


    if 'Churn' in dataframe.columns:
        dataframe['Churn'] = dataframe['Churn'].map({"Yes": 1, "No": 0}).astype(int)

    return dataframe

def get_preprocessor() -> ColumnTransformer:
    columns_for_ohe = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    preprocessor = ColumnTransformer(
        transformers=[('ohe', OneHotEncoder(sparse_output=False, drop='first'), columns_for_ohe)],
        remainder='passthrough'
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor