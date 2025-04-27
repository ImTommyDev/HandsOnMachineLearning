# -------------------------------------------------------------------------
# IMPORTAR LIBRERÍAS NECESARIAS
# -------------------------------------------------------------------------

import matplotlib.pyplot as plt   # Librería para crear gráficos
import numpy as np                # Librería para trabajar con arrays numéricos
import pandas as pd               # Librería para manejar datos en tablas (DataFrames)
from sklearn.linear_model import LinearRegression  # De scikit-learn, importamos la clase para regresión lineal

# -------------------------------------------------------------------------
# DESCARGAR Y PREPARAR LOS DATOS
# -------------------------------------------------------------------------

# Definimos la URL donde están los datos CSV
data_root = "https://raw.githubusercontent.com/ageron/data/refs/heads/main/"

# Leemos el archivo CSV desde la URL y lo guardamos en un DataFrame llamado lifesat
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

# Separamos las variables:
# x será la entrada (feature) → PIB per cápita
x = lifesat[["GDP per capita (USD)"]].values  
# Doble corchete [[ ]] porque scikit-learn necesita que X sea una matriz (2D)

# y será la salida (target) → Satisfacción de vida
y = lifesat["Life satisfaction"].values  
# Solo un corchete [] porque Y puede ser un vector (1D)

# -------------------------------------------------------------------------
# VISUALIZAR LOS DATOS ORIGINALES
# -------------------------------------------------------------------------

# Creamos un gráfico de dispersión para ver cómo se relacionan las dos variables
lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
# grid=True añade una cuadrícula para facilitar la lectura

# Establecemos manualmente los límites de los ejes:
# X: desde 23,500 hasta 62,500
# Y: desde 4 hasta 9
plt.axis([23_500, 62_500, 4, 9])

# Añadimos un título al gráfico
plt.title("Life satisfaction vs GDP per capita")

# Mostramos el gráfico en pantalla
plt.show()

# -------------------------------------------------------------------------
# SELECCIONAR Y ENTRENAR UN MODELO
# -------------------------------------------------------------------------

# Creamos un objeto de la clase LinearRegression
model = LinearRegression()

# Entrenamos (ajustamos) el modelo usando los datos X e Y
model.fit(x, y)
# En este proceso, el modelo calcula:
#  - La mejor pendiente (coeficiente) para la línea
#  - El mejor intercepto (punto de cruce con el eje Y)

# -------------------------------------------------------------------------
# MOSTRAR LA FÓRMULA APRENDIDA POR EL MODELO
# -------------------------------------------------------------------------

# Obtenemos la pendiente y el intercepto de la línea
pendiente = model.coef_[0]  # Cuánto sube Y cuando X sube 1 unidad
intercepto = model.intercept_  # Valor de Y cuando X es 0

# Imprimimos la fórmula de la recta
print(f"La fórmula del modelo aprendido es:")
print(f"Life satisfaction = {pendiente:.5f} * GDP per capita (USD) + {intercepto:.5f}")

# -------------------------------------------------------------------------
# HACER UNA PREDICCIÓN PARA UN NUEVO PAÍS (Chipre)
# -------------------------------------------------------------------------

# Creamos un nuevo array (2D) que representa a Chipre con PIB per cápita de 35,000
X_new = np.array([[35_000]])

# Usamos el modelo para predecir la satisfacción de vida de Chipre
y_new = model.predict(X_new)

# Mostramos la predicción en consola
print("Predicción de Life Satisfaction para Chipre (GDP 35,000 USD):", y_new[0])

# -------------------------------------------------------------------------
# VISUALIZAR TODO EN UN ÚNICO GRÁFICO
# -------------------------------------------------------------------------

# Creamos una nueva figura para el gráfico
plt.figure(figsize=(8,6))  # Tamaño de 8x6 pulgadas

# 1. Dibujamos los puntos originales (datos reales)
plt.scatter(x, y, color='blue', label='Datos reales')

# 2. Dibujamos la línea de regresión aprendida
x_line = np.linspace(23_500, 62_500, 100).reshape(100, 1)  # 100 puntos entre 23,500 y 62,500
y_line = model.predict(x_line)  # Calculamos las predicciones para esos 100 puntos
plt.plot(x_line, y_line, color='red', linewidth=2, label='Modelo lineal')

# 3. Dibujamos el punto de Chipre
plt.scatter(X_new, y_new, color='green', marker='X', s=200, label='Chipre (predicción)')
# - color='green' → color verde
# - marker='X' → el marcador es una X grande
# - s=200 → tamaño del marcador

# Añadimos etiquetas a los ejes y un título
plt.xlabel("GDP per capita (USD)")
plt.ylabel("Life satisfaction")
plt.title("Life satisfaction vs GDP per capita (con predicción para Chipre)")

# Añadimos cuadrícula para mejor lectura
plt.grid(True)

# Establecemos de nuevo los límites de los ejes
plt.axis([23_500, 62_500, 4, 9])

# Añadimos una leyenda para identificar cada cosa
plt.legend()

# Mostramos el gráfico final
plt.show()
