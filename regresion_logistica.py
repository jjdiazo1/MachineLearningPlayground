import numpy as np
import matplotlib.pyplot as plt

# Paso 1: Cargar los datos
# Asegúrate de que el archivo 'datosNN.txt' está en el mismo directorio que este script
data = np.loadtxt('data/datosNN.txt')

# Extraer las características X
X = data  # X es una matriz de Nx2, donde N es el número de muestras

# Paso 2: Definir los pesos y el sesgo (bias)
w = np.array([-2, 1])  # Vector de pesos w = [-2, 1]^T
b = -1                 # Sesgo b = -1

# Paso 3: Calcular z = w^T x + b para cada muestra
z = np.dot(X, w) + b

# Paso 4: Aplicar la función sigmoide para obtener y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

y = sigmoid(z)

# Paso 5: Clasificar las muestras utilizando un umbral de 0.5
labels = (y >= 0.5).astype(int)

# Paso 6: Generar el scatter plot con código de colores según las etiquetas
plt.figure(figsize=(10, 8))

# Separar las muestras según las etiquetas
class_0 = X[labels == 0]
class_1 = X[labels == 1]

# Graficar las muestras de cada clase con colores diferentes
plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Clase 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Clase 1')

plt.title('Clasificación de datos usando Regresión Logística')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)
plt.show()
