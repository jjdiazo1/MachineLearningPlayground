import numpy as np
import matplotlib.pyplot as plt

# carga
datos = np.loadtxt('greenhouse.txt')
Temperatura = datos[:, 0]  
Humedad = datos[:, 1]      

# temp vs humedad
plt.figure(figsize=(8,6))
plt.scatter(Humedad, Temperatura, color='blue')
plt.xlabel('Humedad Relativa (%)')
plt.ylabel('Temperatura (°C)')
plt.title('Temperatura vs. Humedad Relativa')
plt.show()

# reg lineal
beta1, beta0 = np.polyfit(Humedad, Temperatura, 1)

print(f'Parámetros estimados: β₀ = {beta0:.4f}, β₁ = {beta1:.4f}')

# linea de regresion
plt.figure(figsize=(8,6))
plt.scatter(Humedad, Temperatura, color='blue', label='Datos')
plt.plot(Humedad, beta0 + beta1 * Humedad, color='red', label='Línea de Regresión')
plt.xlabel('Humedad Relativa (%)')
plt.ylabel('Temperatura (°C)')
plt.title('Temperatura vs. Humedad Relativa con Línea de Regresión')
plt.legend()
plt.show()

# redisuos
residuos = Temperatura - (beta0 + beta1 * Humedad)
desviacion_estandar_residuos = np.std(residuos)

# umbral
umbral = 2 * desviacion_estandar_residuos

print(f'Desviación estándar de los residuos: {desviacion_estandar_residuos:.4f}')
print(f'Umbral para anomalías: ±{umbral:.4f}')

# nuevosdatos
datos_nuevos = np.loadtxt('data/datosNuevos.txt')
Temperatura_nueva = datos_nuevos[:, 0]
Humedad_nueva = datos_nuevos[:, 1]

# residuos para los nuevos
residuos_nuevos = Temperatura_nueva - (beta0 + beta1 * Humedad_nueva)

# anomalas o no
anomalias = np.abs(residuos_nuevos) > umbral

# graficofinal
plt.figure(figsize=(8,6))
plt.scatter(Humedad_nueva[~anomalias], Temperatura_nueva[~anomalias], color='green', label='Típico')
plt.scatter(Humedad_nueva[anomalias], Temperatura_nueva[anomalias], color='red', label='Anómalo')
plt.xlabel('Humedad Relativa (%)')
plt.ylabel('Temperatura (°C)')
plt.title('Nuevas Mediciones: Anómalas vs. Típicas')
plt.legend()
plt.show()
