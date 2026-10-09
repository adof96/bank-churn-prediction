from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from betabank.config import RANDOM_STATE


def get_models():
    """Modelos candidatos, sin entrenar."""
    return {
        'arbol': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'arbol_balanceado': DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight='balanced'),
        'regresion_logistica': LogisticRegression(
            random_state=RANDOM_STATE, solver='liblinear', class_weight='balanced'),
        'bosque_aleatorio': RandomForestClassifier(
            n_estimators=150, random_state=RANDOM_STATE,
            class_weight='balanced', min_samples_leaf=3),
    }
