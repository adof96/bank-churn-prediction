from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'datasets' / 'Churn.csv'
FIGURES_DIR = ROOT / 'Figuras'

RANDOM_STATE = 12345
TARGET = 'Exited'

# columnas identificadoras: no aportan informacion al modelo
DROP_COLS = ['RowNumber', 'CustomerId', 'Surname']

NUMERIC = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
           'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# proporciones de train / valid / test
VALID_SIZE = 0.2
TEST_SIZE = 0.2

UPSAMPLE_REPEAT = 4
DOWNSAMPLE_FRACTION = 0.25

F1_GOAL = 0.59
