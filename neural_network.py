import numpy as np
import matplotlib.pyplot as plt

# funciones de activación
def relu(z):
    return np.maximum(0, z)

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# neuronas
def neurona_1(x):
    w = np.array([0.1188, -2.1436])
    b = 0.7065
    return relu(np.dot(w, x) + b)

def neurona_2(x):
    w = np.array([1.6878, 0.2615])
    b = -1.0121
    return relu(np.dot(w, x) + b)

def neurona_3(x):
    w = np.array([-0.4082, -0.8420])
    b = 0
    return relu(np.dot(w, x) + b)

def neurona_4(x):
    w = np.array([1.1433, 1.9843])
    b = -0.1435
    return relu(np.dot(w, x) + b)

def neurona_5(x):
    w = np.array([-1.3052, 1.6268])
    b = -2.1021
    return sigmoide(np.dot(w, x) + b)

# fnc de propagación (la que pasa entre las neuronas)
def propagacion_hacia_adelante(x):
    # primera capa
    h1 = neurona_1(x)
    h2 = neurona_2(x)
    
    # sgnda capa
    h3 = neurona_3(np.array([h1, h2]))
    h4 = neurona_4(np.array([h1, h2]))
    
    # capa de salida
    y = neurona_5(np.array([h3, h4]))
    return y

# cargar los datos de entrada
ruta_datos = 'data/datosNN.txt'
datos = np.loadtxt(ruta_datos)

# clasificar cada punto de datos
predicciones = []
for x in datos:
    y = propagacion_hacia_adelante(x)
    etiqueta = 1 if y >= 0.5 else 0
    predicciones.append(etiqueta)

datos = np.array(datos)
predicciones = np.array(predicciones)

plt.figure(figsize=(8, 6))
plt.scatter(datos[predicciones == 0, 0], datos[predicciones == 0, 1], color='blue', label='Clase 0')
plt.scatter(datos[predicciones == 1, 0], datos[predicciones == 1, 1], color='red', label='Clase 1')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Gráfico de Dispersión 2D de Datos Clasificados')
plt.legend()
plt.show()
