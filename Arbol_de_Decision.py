#%%
# Importar Librerias
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

#%%
# Cargar datos
# un array para las imágenes, otro para las etiquetas
data_imgs = np.load('mnistc_images.npy')
data_chrs = np.load('mnistc_labels.npy')[:,np.newaxis]

# %%
# Restringimos el dataset a los dígitos que nos fueron asignados
# Dígitos asociados nuestro grupo
group_digits = [3, 4, 6, 8, 9]

# Filtrar los datos correspondientes a esos dígitos
group_mask = np.isin(data_chrs.flatten(), group_digits)
group_images = data_imgs[group_mask]  # Imágenes de los dígitos seleccionados
group_labels = data_chrs[group_mask]  # Etiquetas de los dígitos seleccionados

# Verificamos las dimensiones de los datos seleccionados y que no se nos haya roto nada
print(f"Cantidad de imágenes seleccionadas: {group_images.shape[0]}")
print(f"Dimensión de cada imagen: {group_images.shape[1:]}")  # 28x28
print(f"Etiquetas seleccionadas: {np.unique(group_labels.flatten())}")

# %%
# Separamos nuestro conjunto de datos en un conjunto de desarrollo y uno de validación, de 80% y 20% del conjunto total, respectivamente.
dev_images, holdout_images, dev_labels, holdout_labels = train_test_split(
    group_images, group_labels, test_size=0.2, random_state=42, stratify=group_labels
)

# Reshapeamos las imagenes bajando la dimension del array para que pueda ser utilizado en las funciones y los métodos que necesitamos
dev_images_reshaped = dev_images.reshape(dev_images.shape[0], -1)
holdout_images_reshaped = holdout_images.reshape(holdout_images.shape[0], -1)


# %%
# Escribimos una funcion para hacer arboles de distintas profundidades y con distintos criterios
# usando k-folding. Devuelve las exactitudes sobre el conjunto de entrenamiento y el conj de validacion por separado.

def entrenar_arboles_kfold(alturas, criterio, X_dev, Y_dev):

    nsplits = len(alturas)
    k_fold = KFold(n_splits=nsplits, shuffle=True, random_state=42)

    promedio_dev = np.zeros(len(alturas))
    promedio_eval = np.zeros(len(alturas))

    for train_index, test_index in k_fold.split(X_dev):
        kf_X_train, kf_X_test = X_dev[train_index], X_dev[test_index]
        kf_Y_train, kf_Y_test = Y_dev[train_index], Y_dev[test_index]

        for j, altura in enumerate(alturas):
            arbol = DecisionTreeClassifier(max_depth=altura, criterion=criterio)
            arbol.fit(kf_X_train, kf_Y_train)


            pred_dev = arbol.predict(kf_X_train)
            pred_eval = arbol.predict(kf_X_test)


            exactitud_dev = metrics.accuracy_score(kf_Y_train, pred_dev)
            exactitud_eval = metrics.accuracy_score(kf_Y_test, pred_eval)


            promedio_dev[j] += exactitud_dev
            promedio_eval[j] += exactitud_eval

    # acá promediamos
    promedio_dev /= nsplits
    promedio_eval /= nsplits

    return {
        "promedio_dev": promedio_dev,
        "promedio_eval": promedio_eval
    }

#%%
# Entrenamos un espectro amplio de alturas, primero con criterio gini, luego con entropía.
alturas = [1, 2, 3, 5, 10, 15, 20, 25, 30]
resultados_gini = entrenar_arboles_kfold(alturas, 'gini', dev_images_reshaped, dev_labels)

#%%
resultados_entropia = entrenar_arboles_kfold(alturas, 'entropy', dev_images_reshaped, dev_labels)

#%%
# Graficamos las exactitudes en funcion de la profundidad, separando exactitud en validacion y en entrenamiento

def grafico_profundidades(resultados_prof, alturas_prof):
    plt.figure(figsize=(10, 6))

    # Graficar la curva de desarrollo
    plt.plot(alturas_prof, resultados_prof["promedio_dev"], label="Exactitud en desarrollo", marker='o', linestyle='--', color='blue')

    # Graficar la curva de evaluación
    plt.plot(alturas_prof, resultados_prof["promedio_eval"], label="Exactitud en evaluación", marker='o', linestyle='-', color='red')

    plt.title("Curvas de desarrollo y evaluación")
    plt.xlabel("Profundidad del árbol")
    plt.ylabel("Exactitud promedio")
    plt.xticks(alturas_prof)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

#%%
grafico_profundidades(resultados_gini, alturas)
grafico_profundidades(resultados_entropia, alturas)

#%%
# entrenamos arboles con las alturas más prometedoras
alturas_nuevas = [7, 8, 9, 10, 11, 12]

resultados_gini_alturas_nuevas = entrenar_arboles_kfold(alturas_nuevas, 'gini', dev_images_reshaped, dev_labels)
resultados_entropia_alturas_nuevas = entrenar_arboles_kfold(alturas_nuevas, 'entropy', dev_images_reshaped, dev_labels)

#%%
grafico_profundidades(resultados_gini_alturas_nuevas, alturas_nuevas)
grafico_profundidades(resultados_entropia_alturas_nuevas, alturas_nuevas)

#%%
# Entrenamos un arbol con altura definida en 10 y con el metodo Gini, para luego evaluarlo en el conj de evaluacion.
# Luego lo usamos para predecir las clases en el conjunto hold out

arbol_elegido = DecisionTreeClassifier(criterion='gini', max_depth= 10)
arbol_elegido.fit(dev_images_reshaped, dev_labels)

#%%
# Predecimos y creamos la matriz de confusion a partir del arbol
prediccion_holdout = arbol_elegido.predict(holdout_images_reshaped)

holdout_labels = holdout_labels.flatten()  # Convertir a unidimensional
clases_interes = [3, 4, 6, 8, 9]

mask = np.isin(holdout_labels, clases_interes)
holdout_labels_filtrados = holdout_labels[mask]
predicciones_filtradas = prediccion_holdout[mask]

matriz_de_confusion = confusion_matrix(holdout_labels_filtrados, predicciones_filtradas, labels=clases_interes)

grafico_mdc = ConfusionMatrixDisplay(confusion_matrix=matriz_de_confusion, display_labels=clases_interes)

grafico_mdc.plot(cmap="viridis")
plt.title("Matriz de Confusión")
plt.show()

#%% 
#Computamos métricas relevantes para evaluar el desempeño de nuestro modelo
print(classification_report(holdout_labels, prediccion_holdout, labels=clases_interes, target_names=[str(x) for x in clases_interes]))
