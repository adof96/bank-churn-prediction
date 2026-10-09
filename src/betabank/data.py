import pandas as pd

from betabank.config import DATA_PATH, DROP_COLS


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def clean(df):
    """Elimina filas sin Tenure y las columnas identificadoras."""
    return df.dropna(subset=['Tenure']).drop(columns=DROP_COLS)
