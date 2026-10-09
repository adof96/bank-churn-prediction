
# %% [markdown]
# # Introduccion
# Los clientes de Beta Bank se están yendo, cada mes, poco a poco. Los banqueros descubrieron que es más barato salvar a los clientes existentes que atraer nuevos.
#
# Necesitamos predecir si un cliente dejará el banco pronto. Tenemos los datos sobre el comportamiento pasado de los clientes y la terminación de contratos con el banco.
#
# Crearemos un modelo con el máximo valor F1 posible. Necesitamos un valor F1 de al menos 0.59. Verificaremos F1 para el conjunto de prueba.
#
# Además, debemos medir la métrica AUC-ROC y compararla con el valor F1
#
# La logica del proyecto vive en el paquete `src/betabank`; este notebook la usa para contar el analisis paso a paso.

# %% [markdown]
# ## Inicializacion y carga de datos

# %%
from sklearn.metrics import confusion_matrix

from betabank.config import DOWNSAMPLE_FRACTION, F1_GOAL, TARGET, UPSAMPLE_REPEAT
from betabank.data import clean, load_data
from betabank.evaluation import compare, evaluate
from betabank.features import prepare
from betabank.models import get_models
from betabank.plots import plot_class_balance, plot_roc
from betabank.sampling import downsample, upsample

# %%
df = load_data()
df.info()
df.head()


# %% [markdown]
# ## Estandarizacion de datos
# Primero es claro que la columna Tenure cuenta con valores ausentes los cuales deben de ser tratados antes de continuar.
# Ademas, `RowNumber`, `CustomerId` y `Surname` son identificadores: no dicen nada sobre si el cliente se ira y `Surname` generaria miles de columnas al codificarla, asi que las eliminamos.

# %%
print(100 * df.isna().sum() / df.shape[0])

# %%
df = clean(df)
print(100 * df.isna().sum() / df.shape[0])


# %% [markdown]
# ### Codificacion OHE, division y escalado de caracteristicas
# Dividimos en entrenamiento (60%), validacion (20%) y prueba (20%), estratificando por la clase objetivo. El escalador se ajusta solo con entrenamiento.

# %%
(features_train, features_valid, features_test,
 target_train, target_valid, target_test) = prepare(df)

print(features_train.shape, features_valid.shape, features_test.shape)


# %% [markdown]
# ## Examinar el equilibrio de clases

# %%
plot_class_balance(df[TARGET])
print(df[TARGET].value_counts(normalize=True))

# %% [markdown]
# Alrededor del 20% de los clientes han abandonado el banco: las clases estan desequilibradas.

# %% [markdown]
# ### Prueba de consistencia
# Entrenamos un arbol de decision sin tener en cuenta el desequilibrio y lo comparamos con un modelo constante que siempre predice 0.

# %%
models = get_models()
tree = models['arbol'].fit(features_train, target_train)
tree_metrics = evaluate(tree, features_valid, target_valid)

print('Exactitud del arbol:          ', tree_metrics['accuracy'])
print('Exactitud del modelo constante:', (target_valid == 0).mean())

# %% [markdown]
# La exactitud del arbol es similar a la del modelo constante, que siempre elige la clase mas frecuente. La exactitud no es una buena metrica cuando hay desequilibrio de clases.

# %% [markdown]
# ### Evaluacion del modelo

# %%
print(confusion_matrix(target_valid, tree.predict(features_valid)))
print('Recall:   ', tree_metrics['recall'])
print('Precision:', tree_metrics['precision'])
print('F1:       ', tree_metrics['f1'])

# %% [markdown]
# El modelo deja escapar a muchos clientes que si se van. Necesitamos un F1 de al menos 0.59, por lo que hay que corregir el desequilibrio.


# %% [markdown]
# ## Mejora de la calidad del modelo
# Probamos cada modelo (arbol, arbol con pesos balanceados, regresion logistica y bosque aleatorio) con tres conjuntos de entrenamiento: original, sobremuestreado y submuestreado. Todos se evaluan en el conjunto de validacion.

# %%
datasets = {
    'original': (features_train, target_train),
    'sobremuestreo': upsample(features_train, target_train, UPSAMPLE_REPEAT),
    'submuestreo': downsample(features_train, target_train, DOWNSAMPLE_FRACTION),
}
results, fitted = compare(models, datasets, features_valid, target_valid)
results.round(4)

# %% [markdown]
# El bosque aleatorio supera claramente a los demas modelos con cualquier conjunto de entrenamiento. La mejor combinacion es el bosque aleatorio con sobremuestreo (F1 ≈ 0.606 en validacion). El submuestreo aumenta el recall, pero baja tanto la precision que el F1 empeora. La regresion logistica y los arboles no llegan a 0.5.

# %% [markdown]
# ## Prueba final del mejor modelo

# %%
best = results.iloc[0]
best_model = fitted[(best['modelo'], best['muestreo'])]
test_metrics = evaluate(best_model, features_test, target_test)

print(f"Mejor modelo: {best['modelo']} ({best['muestreo']})")
for name, value in test_metrics.items():
    print(f'{name:<10} {value:.4f}')
print('Objetivo alcanzado:', test_metrics['f1'] >= F1_GOAL)

# %% [markdown]
# ## Curva Roc

# %%
plot_roc(best_model, features_test, target_test)

# %% [markdown]
# En el conjunto de prueba el bosque aleatorio con sobremuestreo obtiene F1 ≈ 0.635, por encima del objetivo de 0.59, y AUC-ROC ≈ 0.855.
#
# El AUC-ROC es bastante mayor que el F1 porque mide la capacidad del modelo para ordenar a los clientes por riesgo con todos los umbrales posibles. El F1 solo evalua el umbral de 0.5. Un AUC de 0.855 indica que el modelo distingue bien entre clientes que se van y clientes que se quedan, aunque aun esta lejos del caso perfecto (1.0).
#
# Hemos trabajado con diferentes modelos y es importante no quedarse con una sola opcion. Tambien hay que revisar los datos antes de modelar: por ejemplo, eliminar las columnas identificadoras que no aportan informacion al modelo.
