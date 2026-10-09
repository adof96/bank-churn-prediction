"""Pipeline completo: carga, preparacion, comparacion de modelos y evaluacion en test."""
import matplotlib
matplotlib.use('Agg')

from betabank.config import (DOWNSAMPLE_FRACTION, F1_GOAL, FIGURES_DIR,
                             UPSAMPLE_REPEAT)
from betabank.data import clean, load_data
from betabank.evaluation import compare, evaluate
from betabank.features import prepare
from betabank.models import get_models
from betabank.plots import plot_roc
from betabank.sampling import downsample, upsample


def main():
    df = clean(load_data())
    (features_train, features_valid, features_test,
     target_train, target_valid, target_test) = prepare(df)

    datasets = {
        'original': (features_train, target_train),
        'sobremuestreo': upsample(features_train, target_train, UPSAMPLE_REPEAT),
        'submuestreo': downsample(features_train, target_train, DOWNSAMPLE_FRACTION),
    }
    results, fitted = compare(get_models(), datasets, features_valid, target_valid)
    print('Resultados en validacion:')
    print(results.round(4).to_string())

    best = results.iloc[0]
    best_model = fitted[(best['modelo'], best['muestreo'])]
    test_metrics = evaluate(best_model, features_test, target_test)

    print(f"\nMejor modelo: {best['modelo']} ({best['muestreo']})")
    print('Metricas en test:')
    for name, value in test_metrics.items():
        print(f'  {name:<10} {value:.4f}')

    if test_metrics['f1'] < F1_GOAL:
        print(f'\nATENCION: F1 en test por debajo del objetivo de {F1_GOAL}')

    plot_roc(best_model, features_test, target_test,
             path=FIGURES_DIR / 'Roc_curve.png')


if __name__ == '__main__':
    main()
