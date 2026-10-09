import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def evaluate(model, features, target):
    """F1, AUC-ROC, recall, precision y exactitud de un modelo ya entrenado."""
    pred = model.predict(features)
    proba = model.predict_proba(features)[:, 1]
    return {
        'f1': f1_score(target, pred),
        'auc_roc': roc_auc_score(target, proba),
        'recall': recall_score(target, pred),
        'precision': precision_score(target, pred),
        'accuracy': accuracy_score(target, pred),
    }


def compare(models, datasets, features_valid, target_valid):
    """Entrena cada modelo con cada conjunto de entrenamiento y lo evalua en validacion.

    models: dict nombre -> estimador sin entrenar
    datasets: dict nombre -> (features, target)

    Devuelve la tabla de resultados ordenada por F1 y un dict
    (modelo, muestreo) -> estimador entrenado.
    """
    rows = []
    fitted = {}
    for data_name, (features, target) in datasets.items():
        for model_name, model in models.items():
            model = clone(model).fit(features, target)
            fitted[(model_name, data_name)] = model
            rows.append({'modelo': model_name, 'muestreo': data_name,
                         **evaluate(model, features_valid, target_valid)})

    results = (pd.DataFrame(rows)
               .sort_values('f1', ascending=False)
               .reset_index(drop=True))
    return results, fitted
