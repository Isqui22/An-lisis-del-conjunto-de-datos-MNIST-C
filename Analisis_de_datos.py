# Este archivo contiene las funciones utilizadas en el informe del TP.
# En este archivo se trata específicamente la parte del análisis de los datos.

#%%
# Importar Librerias
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#%% 
# Cargar datos
# un array para las imágenes, otro para las etiquetas
data_imgs = np.load('mnistc_images.npy')
data_chrs = np.load('mnistc_labels.npy')[:,np.newaxis]

#%% exploramos algunas características del dataset

# 1. Cantidad de datos
num_samples = data_imgs.shape[0]
img_height, img_width = data_imgs.shape[1], data_imgs.shape[2]
print(f"Cantidad de datos: {num_samples}")
print(f"Dimensiones de cada imagen: {img_height}x{img_width}")

# 2. Cantidad de clases en la variable de interés (cantidad de dígitos distintos)
unique_classes = np.unique(data_chrs)
num_classes = len(unique_classes)
print(f"Cantidad de clases (dígitos): {num_classes}")
print(f"Clases de dígitos: {unique_classes}")

# 3. Otras características
# Frecuencia de cada clase
class_counts = np.unique(data_chrs, return_counts=True)
print(f"Frecuencia de cada clase (dígito): {dict(zip(class_counts[0], class_counts[1]))}")

# Rango de valores en las imágenes
min_value = data_imgs.min()
max_value = data_imgs.max()
print(f"Rango de valores en las imágenes: {min_value} a {max_value}")

#%% PRECAUCIÓN: Toma mucho tiempo de carga
# miramos si hay duplicados 

# Crear un conj para almacenar las imágenes únicas
unique_images = []
# Variable para contar los duplicados
duplicates_count = 0

# Recorrer el subconjunto de imágenes
for img in data_imgs:
    # Verificar si la imagen ya está en unique_images
    if any(np.array_equal(img, unique_img) for unique_img in unique_images):
        duplicates_count += 1
    else:
        unique_images.append(img)
        
# Mostrar el número de imágenes duplicadas en el conjunto
print(f"Número de imágenes duplicadas en el conjunto: {duplicates_count}")
print(f"Número de imágenes únicas en el conjunto: {len(unique_images)}")

#%%
# Hacemos gráficos que nos ayuden a explorar los datos
def graficos(data_imagenes, data_labels, n_digit, numero):
  for i in range(n_digit):
    image_array = data_imgs[i,:,:,0]
    image_label = data_chrs[i]
    if (image_label == [numero]):
  # Ploteo el grafico
      plt.figure(figsize=(10,8))
      plt.imshow(image_array, cmap='gray')
      plt.title('caracter: ' + str(image_label))
      plt.axis('off')
      plt.show()
      
print(graficos(data_imgs,data_chrs,24,3))

#%%
# Grafico de Mapa de calor
def superposición_de_calor(n):
    digit_images = data_imgs[data_chrs.flatten() == n]
    #Promedio de imagenes
    average_image = np.mean(digit_images,axis = 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(average_image, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Valor del píxel promedio")  
    plt.title("Imagen promedio de los píxeles (MNIST-C Blur)")
    plt.axis('off')  
    plt.show()

print(superposición_de_calor(7))

#%%
#Grafico para comparar imagenes de un mismo numero
def grafico_numeros_iguales(digit):
 
  digit_images = data_imgs[data_chrs.flatten() == digit]

  
  fig, axes = plt.subplots(2, 5, figsize=(10, 5))
  for i, ax in enumerate(axes.flat):
      if i < len(digit_images):
          ax.imshow(digit_images[i], cmap='gray')
          ax.set_title(f"Dígito: {digit}")
          ax.axis('off')
  plt.show()

grafico_numeros_iguales(4)
grafico_numeros_iguales(1)

#%%
# Función para calcular la distancia promedio entre dos conjuntos de imágenes
def calculate_average_distance(images1, images2):
    distances = []
    for img1 in images1:
        for img2 in images2:
            distance = euclidean(img1.flatten(), img2.flatten())
            distances.append(distance)
    return np.mean(distances)

digit_0 = data_imgs[data_chrs.flatten() == 0]
digit_1 = data_imgs[data_chrs.flatten() == 1]
digit_5 = data_imgs[data_chrs.flatten() == 5]
digit_6 = data_imgs[data_chrs.flatten() == 6]

average_distance_0_1 = calculate_average_distance(digit_0, digit_1)
average_distance_5_6 = calculate_average_distance(digit_5, digit_6)
average_distance_1_5 = calculate_average_distance(digit_1, digit_5)

# Mostrar la distancia promedio
print(f"Distancia Euclidiana promedio entre todas las imágenes de 0 y 1: {average_distance_0_1}")
print(f"Distancia Euclidiana promedio entre todas las imágenes de 5 y 6: {average_distance_5_6}")
print(f"Distancia Euclidiana promedio entre todas las imágenes de 1 y 5: {average_distance_1_5}")

#%% Representacón PCA
# agregamos otros dos digitos para mas adelante
digit_7 = data_imgs[data_chrs.flatten() == 7]
digit_8 = data_imgs[data_chrs.flatten() == 8]

# Aplanamos las imágenes para que cada imagen sea un vector de 784 dimensiones
images = np.concatenate([digit_0, digit_1, digit_5, digit_6])

# Aplicamos PCA para reducir la dimensionalidad a 2 dimensiones
pca = PCA(n_components=2)
reduced_images = pca.fit_transform(images.reshape(len(images), -1))

# Crear el gráfico de dispersión
plt.figure(figsize=(8, 6))
plt.scatter(reduced_images[:len(digit_0), 0], reduced_images[:len(digit_0), 1], color='blue', label='Dígito 0')
plt.scatter(reduced_images[len(digit_0):len(digit_0) + len(digit_1), 0], reduced_images[len(digit_0):len(digit_0) + len(digit_1), 1], color='red', label='Dígito 1')
plt.scatter(reduced_images[len(digit_0) + len(digit_1):len(digit_0) + len(digit_1) + len(digit_5), 0], reduced_images[len(digit_0) + len(digit_1):len(digit_0) + len(digit_1) + len(digit_5), 1], color='green', label='Dígito 5')
plt.scatter(reduced_images[len(digit_0) + len(digit_1) + len(digit_5):, 0], reduced_images[len(digit_0) + len(digit_1) + len(digit_5):, 1], color='purple', label='Dígito 6')

plt.title('Representación PCA de las Imágenes de los Dígitos 0, 1, 5 y 6')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()

plt.show()