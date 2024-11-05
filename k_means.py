import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

datos = np.loadtxt('data/coordenadasPacientes.txt')

plt.figure(figsize=(10, 8))
plt.scatter(datos[:, 0], datos[:, 1], s=10, color='blue')
plt.title('Gráfico de Dispersión de Ubicaciones de Pacientes')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(datos)

etiquetas = kmeans.labels_
centros = kmeans.cluster_centers_

plt.figure(figsize=(10, 8))
colores = ['red', 'green', 'blue', 'purple']
for i in range(4):
    puntos_cluster = datos[etiquetas == i]
    plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], s=10, color=colores[i], label=f'Cluster {i+1}')

plt.scatter(centros[:, 0], centros[:, 1], s=200, c='yellow', marker='X', label='Centros')

plt.title('Clustering K-means de Ubicaciones de Pacientes')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend()
plt.grid(True)
plt.show()

kmeans_aleatorio = KMeans(n_clusters=4, init='random', random_state=42)
kmeans_aleatorio.fit(datos)

etiquetas_aleatorio = kmeans_aleatorio.labels_
centros_aleatorio = kmeans_aleatorio.cluster_centers_

plt.figure(figsize=(10, 8))
for i in range(4):
    puntos_cluster = datos[etiquetas_aleatorio == i]
    plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], s=10, color=colores[i], label=f'Cluster {i+1}')

plt.scatter(centros_aleatorio[:, 0], centros_aleatorio[:, 1], s=200, c='yellow', marker='X', label='Centros')

plt.title('Clustering K-means con Inicialización Aleatoria')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend()
plt.grid(True)
plt.show()
