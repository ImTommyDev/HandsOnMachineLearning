import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download and prepare the data
data_root = "https://raw.githubusercontent.com/ageron/data/refs/heads/main/" 
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv") #Create a DataFrame from the CSV file
x = lifesat[["GDP per capita (USD)"]].values # x is an array with the data of the GDP per capita column
# The double brackets are used to keep the data as a 2D array, which is required by sklearn
y = lifesat["Life satisfaction"].values # y is an array with the data of the Life satisfaction column
# .values convierte los pandas Series en arrays de NumPy, que luego son más fáciles de usar con scikit-learn.

# Visualize the data
lifesat.plot(kind = 'scatter', grid = True, x = "GDP per capita (USD)", y = "Life satisfaction")
# Crea un gráfico de dispersión (scatter plot), donde:
# El eje x es el GDP per capita (USD).
# El eje y es la Life satisfaction.
# grid=True añade una cuadrícula al fondo del gráfico para facilitar la lectura.
plt.axis([23_500,62_500,4,9]) #plt.axis([xmin, xmax, ymin, ymax]) establece manualmente los rangos que quiero que se vean en el gráfico.
plt.title("Life satisfaction vs GDP per capita")
plt.show()

#Select a linear model
model = LinearRegression() #Crea un objeto de la clase LinearRegression, que es el modelo de regresión lineal de scikit-learn.

#Train the model
model.fit(x, y) #Entrena el modelo con los datos de GDP per capita (x) y Life satisfaction (y).
# El modelo aprende a predecir y (Life satisfaction) a partir de x (GDP per capita).
# ¿Qué significa "aprender" aquí?
    # Encuentra la mejor recta (una línea) que se ajusta a los puntos del scatter plot.
    # Matemáticamente, calcula dos cosas:
        # La pendiente de la línea (cuánto sube por cada dólar extra de GDP).
        # El intercepto (por dónde cruza el eje Y).




#Make a prediction for Cyprus
X_new = np.array([[35_000]]) #Crea un nuevo array con el GDP(PIB) per capita de Chipre (35,000 USD).
#EXPLICACIÓN DE ESTO CON IA:
# Creo un nuevo dato (X_new) que es un país hipotético con un PIB per cápita de 35,000 USD.
# No he leído datos de Chipre, ni el modelo "sabe" que eso es Chipre.
# Simplemente le digo: "Oye modelo, si un país tiene 35,000 USD de GDP per capita, ¿cuál sería su satisfacción de vida según la recta que aprendiste?"
# Entonces el modelo:
# Toma 35,000, lo mete en la fórmula de la línea recta que aprendió, y me devuelve un valor estimado de satisfacción de vida.
# La predicción es un número entre 4 y 9, que es el rango de satisfacción de vida que se ve en el gráfico.

print(model.predict(X_new)) #Predice la Life satisfaction para Chipre usando el modelo entrenado.
#La consola me devuelve la predicción de satisfacción de vida para Chipre ([6.12166432])