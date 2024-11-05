import numpy as np
import matplotlib.pyplot as plt

datos = np.loadtxt('data/datosToy.txt')
x = datos[:, 0]
y = datos[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gráfico de Dispersión de y vs x')
plt.show()

beta1_lineal, beta0_lineal = np.polyfit(x, y, 1)

print(f'Parámetros estimados para el modelo lineal:')
print(f'β₀ = {beta0_lineal:.4f}')
print(f'β₁ = {beta1_lineal:.4f}')

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='gray', s=10, label='Puntos de Datos')
plt.plot(x, beta0_lineal + beta1_lineal * x, color='red', label='Ajuste Lineal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste Lineal: y = β₀ + β₁x')
plt.legend()
plt.show()

coeficientes_cuadratico = np.polyfit(x, y, 2)
beta2_cuadratico, beta1_cuadratico, beta0_cuadratico = coeficientes_cuadratico

print(f'\nParámetros estimados para el modelo cuadrático:')
print(f'β₀ = {beta0_cuadratico:.4f}')
print(f'β₁ = {beta1_cuadratico:.4f}')
print(f'β₂ = {beta2_cuadratico:.4f}')

x_ajuste = np.linspace(np.min(x), np.max(x), 500)
y_ajuste_cuadratico = beta0_cuadratico + beta1_cuadratico * x_ajuste + beta2_cuadratico * x_ajuste**2

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='gray', s=10, label='Puntos de Datos')
plt.plot(x_ajuste, y_ajuste_cuadratico, color='green', label='Ajuste Cuadrático')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste Cuadrático: y = β₀ + β₁x + β₂x²')
plt.legend()
plt.show()

coeficientes_cubico = np.polyfit(x, y, 3)
beta3_cubico, beta2_cubico, beta1_cubico, beta0_cubico = coeficientes_cubico

print(f'\nParámetros estimados para el modelo cúbico:')
print(f'β₀ = {beta0_cubico:.4f}')
print(f'β₁ = {beta1_cubico:.4f}')
print(f'β₂ = {beta2_cubico:.4f}')
print(f'β₃ = {beta3_cubico:.4f}')

x_ajuste = np.linspace(np.min(x), np.max(x), 500)
y_ajuste_cubico = beta0_cubico + beta1_cubico * x_ajuste + beta2_cubico * x_ajuste**2 + beta3_cubico * x_ajuste**3

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='gray', s=10, label='Puntos de Datos')
plt.plot(x_ajuste, y_ajuste_cubico, color='orange', label='Ajuste Cúbico')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste Cúbico: y = β₀ + β₁x + β₂x² + β₃x³')
plt.legend()
plt.show()

# Ajuste de cuarto grado
coeficientes_cuarto = np.polyfit(x, y, 4)
beta4_cuarto, beta3_cuarto, beta2_cuarto, beta1_cuarto, beta0_cuarto = coeficientes_cuarto
print(f'\nParámetros estimados para el modelo de cuarto grado:')
print(f'β₀ = {beta0_cuarto:.4f}')
print(f'β₁ = {beta1_cuarto:.4f}')
print(f'β₂ = {beta2_cuarto:.4f}')
print(f'β₃ = {beta3_cuarto:.4f}')
print(f'β₄ = {beta4_cuarto:.4f}')

y_ajuste_cuarto = (beta0_cuarto + beta1_cuarto * x_ajuste + beta2_cuarto * x_ajuste**2 +
                   beta3_cuarto * x_ajuste**3 + beta4_cuarto * x_ajuste**4)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='gray', s=10, label='Puntos de Datos')
plt.plot(x_ajuste, y_ajuste_cuarto, color='purple', label='Ajuste Cuarto Grado')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste de Cuarto Grado: y = β₀ + β₁x + β₂x² + β₃x³ + β₄x⁴')
plt.legend()
plt.show()

# Ajuste de quinto grado
coeficientes_quinto = np.polyfit(x, y, 5)
beta5_quinto, beta4_quinto, beta3_quinto, beta2_quinto, beta1_quinto, beta0_quinto = coeficientes_quinto
print(f'\nParámetros estimados para el modelo de quinto grado:')
print(f'β₀ = {beta0_quinto:.4f}')
print(f'β₁ = {beta1_quinto:.4f}')
print(f'β₂ = {beta2_quinto:.4f}')
print(f'β₃ = {beta3_quinto:.4f}')
print(f'β₄ = {beta4_quinto:.4f}')
print(f'β₅ = {beta5_quinto:.4f}')

y_ajuste_quinto = (beta0_quinto + beta1_quinto * x_ajuste + beta2_quinto * x_ajuste**2 +
                   beta3_quinto * x_ajuste**3 + beta4_quinto * x_ajuste**4 + beta5_quinto * x_ajuste**5)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='gray', s=10, label='Puntos de Datos')
plt.plot(x_ajuste, y_ajuste_quinto, color='brown', label='Ajuste Quinto Grado')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste de Quinto Grado: y = β₀ + β₁x + β₂x² + β₃x³ + β₄x⁴ + β₅x⁵')
plt.legend()
plt.show()