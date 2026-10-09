import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def plot_class_balance(target, path=None):
    ax = target.value_counts(normalize=True).plot(kind='bar')
    ax.set_title('Frecuencia de clases')
    _finish(path)


def plot_roc(model, features, target, path=None):
    proba = model.predict_proba(features)[:, 1]
    fpr, tpr, _ = roc_curve(target, proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC')
    _finish(path)


def _finish(path):
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    if plt.get_backend().lower() == 'agg':
        plt.close()
    else:
        plt.show()
