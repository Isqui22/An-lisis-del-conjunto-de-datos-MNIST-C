# Analisis-MNIST-C

## Introducción
En el presente proyecto se trabajará con el conjunto de datos de imágenes (de 28x28 píxeles en
escala de grises con valores 0-255) denominado MNIST-C en su versión “Motion Blur”. Cada imagen del
set de datos representa un dígito escrito a mano entre 0 y 9, ambos inclusive. Se trata de una versión
ligeramente corrompida de un dataset muy famoso llamado [MNIST1](https://en.wikipedia.org/wiki/MNIST_database). En la sección Referencias hay links al dataset MNIST-C completo y el trabajo donde lo presentan. Ahí pueden ver el tipo de alteración que se eligió y todas las que hay disponibles.

## Proyecto
Para este proyecto, nos proponemos construir un modelo de clasificación basado en
árboles de decisión para lograr identificar que dígito númerico (3, 4, 6, 8 o 9) corresponde
a una imágen escrita a mano. Nuestro objetivo es analizar y encontrar cuáles son las
características de interés y evaluar el rendimiento del modelo para establecer su capacidad
de generalización.

### Referencias
Dataset completo: https://zenodo.org/records/3239543
Trabajo original donde se presenta el dataset (preprint): https://arxiv.org/abs/1906.02337
