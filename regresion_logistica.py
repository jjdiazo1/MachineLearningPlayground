import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data/datosNN.txt')

X = data  # X es una matriz de Nx2, donde N es el número de muestras

w = np.array([-2, 1])  # Vector de pesos w = [-2, 1]^T
b = -1                 # Sesgo b = -1

z = np.dot(X, w) + b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

y = sigmoid(z)

labels = (y >= 0.5).astype(int)

plt.figure(figsize=(10, 8))

class_0 = X[labels == 0]
class_1 = X[labels == 1]

plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Clase 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Clase 1')

plt.title('Clasificación de datos usando Regresión Logística')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.grid(True)
plt.show()
