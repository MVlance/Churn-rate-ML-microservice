import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame | str:
    if not os.path.exists(file_path):
        raise FileNotFoundError("file not found")

    return pd.read_csv(file_path)