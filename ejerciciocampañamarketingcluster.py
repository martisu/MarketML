# -*- coding: utf-8 -*-
"""EjercicioCampañaMarketingCluster.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AElbGCQ_D_uEViXOA0QDsoZdFCnZWNCW

### Preprocesamiento y clustering con datos de campaña de marketing

#### Inspirado de https://www.kaggle.com/code/karnikakapoor/customer-segmentation-clustering

### Análisis exploratorio
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
file_path = 'marketing_campaign.csv'
data = pd.read_csv(file_path, delimiter=';')

# Inspección Inicial
print("Información del dataset:")
print(data.info())

print("\nResumen estadístico:")
print(data.describe())

print("\nValores nulos por columna:")
print(data.isnull().sum())

# Análisis Univariado

# Distribución de variables numéricas
num_columns = data.select_dtypes(include=['float64', 'int64']).columns

for col in num_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()

# Distribución de variables categóricas
cat_columns = data.select_dtypes(include=['object']).columns

for col in cat_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=data[col], order=data[col].value_counts().index)
    plt.title(f'Distribución de {col}')
    plt.xlabel('Frecuencia')
    plt.ylabel(col)
    plt.show()

# Cantidad de datos por categoría en 'Marital_Status'
marital_status_counts = data['Marital_Status'].value_counts()

# Cantidad de datos por categoría en 'Education'
education_counts = data['Education'].value_counts()

# Mostrar los resultados
print("Cantidad de datos por categoría en 'Marital_Status':")
print(marital_status_counts)

print("\nCantidad de datos por categoría en 'Education':")
print(education_counts)

#Análisis Bivariado
# Relación entre ingresos y el número de compras de vino
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Income', y='MntWines', data=data)
plt.title('Relación entre Ingresos y Compras de Vino')
plt.xlabel('Ingresos')
plt.ylabel('Compras de Vino')
plt.show()

# Relación entre el estado civil y el ingreso promedio
plt.figure(figsize=(10, 6))
sns.boxplot(x='Marital_Status', y='Income', data=data)
plt.title('Ingreso promedio por Estado Civil')
plt.xlabel('Estado Civil')
plt.ylabel('Ingresos')
plt.show()

"""### Limpieza de datos"""

# Histograma de la distribución de 'Income' antes de la imputación
plt.figure(figsize=(10, 6))
sns.histplot(data['Income'], kde=True, bins=30)
plt.title('Distribución de Income antes de la imputación')
plt.xlabel('Income')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de Densidad de la distribución de 'Income'
plt.figure(figsize=(10, 6))
sns.kdeplot(data['Income'].dropna(), fill=True)
plt.title('Distribución de Densidad de Income antes de la imputación')
plt.xlabel('Income')
plt.ylabel('Densidad')
plt.show()

# Imputación con la Media
data_mean = data.copy()
data_mean['Income'].fillna(data_mean['Income'].mean(), inplace=True)

# Imputación con la Mediana
data_median = data.copy()
data_median['Income'].fillna(data_median['Income'].median(), inplace=True)

# Imputación con la Moda
data_mode = data.copy()
data_mode['Income'].fillna(data_mode['Income'].mode()[0], inplace=True)

# Imputación con un Valor Fijo (por ejemplo, 0)
data_fixed = data.copy()
data_fixed['Income'].fillna(0, inplace=True)

# Gráfico de Densidad de la distribución de 'Income'
# Después de imputar con la media
plt.figure(figsize=(10, 6))
sns.kdeplot(data_mean['Income'].dropna(), fill=True)
plt.title('Distribución de Densidad de Income después de la imputación con la media')
plt.xlabel('Income')
plt.ylabel('Densidad')
plt.show()

# Después de imputar con la Mediana
plt.figure(figsize=(10, 6))
sns.kdeplot(data_median['Income'].dropna(), fill=True)
plt.title('Distribución de Densidad de Income después de la imputación con la Mediana')
plt.xlabel('Income')
plt.ylabel('Densidad')
plt.show()

# Después de imputar con la Moda
plt.figure(figsize=(10, 6))
sns.kdeplot(data_mode['Income'].dropna(), fill=True)
plt.title('Distribución de Densidad de Income después de la imputación con la Moda')
plt.xlabel('Income')
plt.ylabel('Densidad')
plt.show()

# Después de imputar con un valor fijo
plt.figure(figsize=(10, 6))
sns.kdeplot(data_fixed['Income'].dropna(), fill=True)
plt.title('Distribución de Densidad de Income después de la imputación un valor fijo')
plt.xlabel('Income')
plt.ylabel('Densidad')
plt.show()

# Imputación por Regresión
# Seleccionar variables para el modelo de regresión
# Aquí se seleccionan variables que podrían estar relacionadas con los ingresos
from sklearn.linear_model import LinearRegression
regression_data = data.dropna(subset=['Income'])
X = regression_data[['Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts']]
y = regression_data['Income']

# Entrenar un modelo de regresión
model = LinearRegression()
model.fit(X, y)

# Predecir los valores faltantes de Income
X_null = data[data['Income'].isnull()][['Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts']]
data_regression = data.copy()
data_regression.loc[data['Income'].isnull(), 'Income'] = model.predict(X_null)

# Después de imputar con una regresión
plt.figure(figsize=(10, 6))
sns.kdeplot(data_regression['Income'].dropna(), fill=True)
plt.title('Distribución de Densidad de Income después de la imputación utilizando regresión')
plt.xlabel('Income')
plt.ylabel('Densidad')
plt.show()

# Eliminar filas donde 'Income' es nulo
data_cleaned = data.dropna(subset=['Income'])

data_cleaned.info()

##Reemplazar los valores nulos de la variable INCOME por los imputados: Elegimos la media
# Calcular la media de la columna 'Income'
income_mean = data['Income'].mean()

# Reemplazar los valores nulos (NaN) por la media
data['Income'].fillna(income_mean, inplace=True)

# Verificar si hay valores nulos en la columna 'Income' después de reemplazar
print(data['Income'].isnull().sum())

##Discretizar la columna fecha Dt_Customer

# Convertir la columna 'Dt_Customer' a formato de fecha
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%Y-%m-%d')

# Obtener la fecha más reciente y la fecha más antigua en los registros
fecha_mas_reciente = data['Dt_Customer'].max()
fecha_mas_antigua = data['Dt_Customer'].min()

# Calcular la antigüedad en días desde la fecha más reciente
data['Customer_Age_Days'] = (fecha_mas_reciente - data['Dt_Customer']).dt.days
# Calcular la antigüedad en días desde la fecha más antigua
#data['Customer_Age_Days_From_Earliest'] = (data['Dt_Customer'] - fecha_mas_antigua).dt.days

# Mostrar las fechas más reciente y más antigua, y las primeras filas del DataFrame con la nueva columna
print("Fecha más reciente:", fecha_mas_reciente)
print("Fecha más antigua:", fecha_mas_antigua)
print(data[['Dt_Customer', 'Customer_Age_Days']].head())

#Calcular edad por cliente

# Obtener el año actual
#anio_actual = pd.to_datetime('today').year

# Calcular la edad
data['Age'] = 2014 - data['Year_Birth']

# Mostrar las primeras filas del DataFrame con la nueva columna de edad
print(data[['Year_Birth', 'Age']].head())

#Calcular total gastado en varias categorías por cliente
#Total gastado en varios items
data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]
# Mostrar las primeras filas del DataFrame con la nueva columna
print(data[['Spent']].head())

#Simplificar las categorías de Marital_Status, y crear solo dos categorías Acompañado o solo
data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})
# Mostrar las primeras filas del DataFrame con la nueva columna
print(data[['Living_With']].head())

#Indicar la cantidad de niños totales
data["Children"]=data["Kidhome"]+data["Teenhome"]
# Mostrar las primeras filas del DataFrame con la nueva columna
print(data[['Children']].head())

#Indicar el número de miembros totales
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]
# Mostrar las primeras filas del DataFrame con la nueva columna
print(data[['Family_Size']].head())

import numpy as np
#Crear una columna indicando si es padre o no
data["Is_Parent"] = np.where(data.Children> 0, 1, 0)
# Mostrar las primeras filas del DataFrame con la nueva columna
print(data[['Is_Parent']].head())

#Segmentar el nivel educativo en tres grupos
data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})
# Mostrar las primeras filas del DataFrame con la nueva columna
print(data[['Education']].head())

#Borrar características redundantes
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
data = data.drop(to_drop, axis=1)
data

data.describe()

from matplotlib import colors
import seaborn as sns
import matplotlib.pyplot as plt

# Configurar paleta de colores personalizada
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})
pallet = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B"]
cmap = colors.ListedColormap(pallet)

# Graficar ingresos, días que han pasado desde la última compra, Antigüedad cliente, edad, gastos totales, es padre?
To_Plot = ["Income", "Recency", "Customer_Age_Days", "Age", "Spent", "Is_Parent"]

print("Gráficos de dispersión de las características seleccionadas")

# Actualizar la paleta de colores para la visualización con hue
plt.figure()
sns.pairplot(data[To_Plot], hue="Is_Parent", palette=sns.color_palette(pallet[:2]))

# Mostrar los gráficos
plt.show()

## Existen valores atípicos en las características de ingresos y edad. Vamos a eliminar valores atípicos
data = data[(data["Age"]<90)]
data = data[(data["Income"]<600000)]
print("Número total de registros que quedan:", len(data))

# Excluir las columnas categóricas
numericData = data.select_dtypes(include=[float, int])

# Calcular la matriz de correlación
correlationMatrixPearson = numericData.corr()

# Mostrar la matriz de correlación
plt.figure(figsize=(20, 20))
sns.heatmap(correlationMatrixPearson, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Matriz de Correlación (Excluyendo Características Categóricas)')
plt.show()

"""### Correlación Pearson"""

# Establecer el umbral de correlación
threshold = 0.85

# Encontrar los pares de variables que superan el umbral
high_correlation_pairs = []

# Función para encontrar pares de variables que superan el umbral
def encontrar_pares_alta_correlacion(correlationMatrixPearson, threshold=0.85):
    high_correlation_pairs = []
    for i in range(len(correlationMatrixPearson.columns)):
        for j in range(i):
            if abs(correlationMatrixPearson.iloc[i, j]) > threshold:
                high_correlation_pairs.append((correlationMatrixPearson.columns[i], correlationMatrixPearson.columns[j], correlationMatrixPearson.iloc[i, j]))
    return high_correlation_pairs

# Encontrar los pares con alta correlación
pares = encontrar_pares_alta_correlacion(correlationMatrixPearson, threshold)

# Imprimir los pares de alta correlación
if pares:
    for pair in pares:
        print(f"Las variables {pair[0]} y {pair[1]} tienen una correlación de {pair[2]:.2f} usando Pearson")
else:
    print("No hay pares de variables con correlación mayor a 0.85 usando Pearson")

"""### Correlación de Spearman

Mide la relación monótona entre variables. Es decir, mide si existe un patrón de crecimiento aunque no de forma proporcional.
Cuando lo usamos con variables continuas, lo que hace es convertir los valores de variables a rangos y medir la relación entre estos rangos. Funciona muy bien con valores outliers.
Interpretación:

1: Correlación positiva perfecta. Si una variable aumenta, la otra también.
-1: Correlación negativa perfecta. Mientras que una variable aumenta, la otra disminuye de manera constante.
0: No existe correlación monótona entre variables.
"""

# Calcular la matriz de correlación de Spearman
correlationMatrixSpearman = numericData.corr(method='spearman')

# Encontrar los pares con alta correlación
pares = encontrar_pares_alta_correlacion(correlationMatrixSpearman, threshold)

# Imprimir los pares de alta correlación
if pares:
    for pair in pares:
        print(f"Las variables {pair[0]} y {pair[1]} tienen una correlación de {pair[2]:.2f} usando Spearman")
else:
    print("No hay pares de variables con correlación mayor a 0.85 usando Spearman")

"""### Correlación de Kendall

Mide la concordancia entre los pares de observaciones, es decir, la frecuencia con la que los valores de las variables cambian en la misma dirección.
El coeficiente de Kendall ofrece una interpretación más exacta en términos de la proporción de pares concordantes o discordantes, por lo que es más exacto al evaluar relaciones monótonas estrictas.
"""

# Calcular la matriz de correlación de Kendall
correlationMatrixKendall = numericData.corr(method='kendall')

# Encontrar los pares con alta correlación
pares = encontrar_pares_alta_correlacion(correlationMatrixKendall, threshold)

# Imprimir los pares de alta correlación
if pares:
    for pair in pares:
        print(f"Las variables {pair[0]} y {pair[1]} tienen una correlación de {pair[2]:.2f} usando Kendall")
else:
    print("No hay pares de variables con correlación mayor a 0.85 usando Kendall")

"""### Preprocesamiento

* Codificación de etiquetas de las características categóricas
* Escalado de las características mediante el escalador estándar
"""

#Obtener variables categóricas
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Variables categóricas del dataset:", object_cols)

from sklearn.preprocessing import LabelEncoder
#Utilizamos el Label Encoding
LE=LabelEncoder()
for i in object_cols:
    data[i]=data[[i]].apply(LE.fit_transform)

print("Ahora las características son numéricas")
data

"""### Escalar los datos"""

#Escalar las variables no binarias y crear un dataset completo
from sklearn.preprocessing import StandardScaler

# Identificar las columnas que son binarias (valores 0 o 1)
binary_columns = data.columns[(data.nunique() == 2) & (data.isin([0, 1]).all())]

# Identificar las columnas no binarias
non_binary_columns = data.columns.difference(binary_columns)

# Escalar solo las columnas no binarias
scaler = StandardScaler()
allDataScaleOnlyNonBinary = data.copy()
allDataScaleOnlyNonBinary[non_binary_columns] = scaler.fit_transform(data[non_binary_columns])

# Verificar el resultado
print(allDataScaleOnlyNonBinary.head())

#Escalar todas las variables, incluidas las binarias
from sklearn.preprocessing import StandardScaler

# Escalar
scaler = StandardScaler()
#data_scaled_all = data
allDataScale = scaler.fit_transform(data)
allDataScale = pd.DataFrame(allDataScale, columns=data.columns)

# Verificar el resultado
print(allDataScale.head())

#Crear una copia de los datos
ds = data.copy()
# Crear un subconjunto del dataframe borrando las características binarias
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response', 'Is_Parent', 'Living_With', 'Complain', 'Response']
ds = ds.drop(cols_del, axis=1)
#Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaledNonBinaryVariables = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
print("All features are now scaled")

"""### K-MEANS

#### Método del codo
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Método del codo para seleccionar el número óptimo de clústeres incluyendo los datos binarios
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(allDataScaleOnlyNonBinary)
    distortions.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8,6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Distorsión (Inertia)')
plt.title('Método del Codo para determinar el número óptimo de clústeres (incluye datos binarios sin escalar)')
plt.show()

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Método del codo para seleccionar el número óptimo de clústeres incluyendo los datos binarios
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(allDataScale)
    distortions.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8,6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Distorsión (Inertia)')
plt.title('Método del Codo para determinar el número óptimo de clústeres escalando las binarias')
plt.show()

# Método del codo para seleccionar el número óptimo de clústeres ignorando datos binarios
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaledNonBinaryVariables)
    distortions.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8,6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Número de Clústeres (k)')
plt.ylabel('Distorsión (Inertia)')
plt.title('Método del Codo para determinar el número óptimo de clústeres ignorando los datos binarios')
plt.show()

"""#### Lloyd's k-Means (clásico k-means de sklearn)"""

# Definir el número de clústeres (k)
k = 3  # Puedes ajustar este valor o usar el método del codo para encontrar el óptimo

# Crear el modelo K-Means
kmeans = KMeans(n_clusters=k, random_state=42)

# Entrenar el modelo con los datos escalados
kmeans.fit(scaledNonBinaryVariables)

# Agregar los clústeres asignados como una nueva columna al DataFrame original
scaledNonBinaryVariables['Cluster'] = kmeans.labels_

# Visualizar los primeros resultados con la columna de clústeres
print(scaledNonBinaryVariables[['Cluster']].head())

# Variables que deseas graficar (puedes cambiar 'Income' y 'MntWines' por las que prefieras)
x_var = 'Income'
y_var = 'MntWines'

# Crear el gráfico de dispersión con los clústeres
plt.figure(figsize=(10, 6))
plt.scatter(scaledNonBinaryVariables[x_var], scaledNonBinaryVariables[y_var], c=scaledNonBinaryVariables['Cluster'], cmap='viridis', marker='o', edgecolor='k')

# Añadir etiquetas y título
plt.xlabel(x_var)
plt.ylabel(y_var)
plt.title(f'Distribución de los clústeres en {x_var} y {y_var}')
plt.colorbar(label='Cluster')

# Mostrar el gráfico
plt.show()

"""### MacQueen’s k-Means
El algoritmo de MacQueen es una variante del k-means que actualiza los centroides después de cada asignación en lugar de hacer todas las asignaciones primero y luego actualiza los centroides. Aunque sklearn no lo incluye de manera nativa, puedes simular el comportamiento modificando un poco el algoritmo clásico.
"""

import numpy as np

# MacQueen's k-means
def macqueen_kmeans(df, n_clusters, max_iter=100):
    # Convertir el DataFrame a un array de numpy
    data = df.to_numpy()

    # Inicializar centroides aleatoriamente
    np.random.seed(42)  # Fijar una semilla para reproducibilidad
    centroids = data[np.random.choice(data.shape[0], n_clusters, replace=False)]

    for _ in range(max_iter):
        for i in range(data.shape[0]):
            # Calcular distancias entre el punto y los centroides
            distances = np.linalg.norm(data[i] - centroids, axis=1)
            # Encontrar el centroide más cercano
            closest = np.argmin(distances)
            # Actualizar el centroide más cercano
            centroids[closest] = centroids[closest] + 0.1 * (data[i] - centroids[closest])

    # Asignar puntos al centroide más cercano
    labels = np.array([np.argmin(np.linalg.norm(data[i] - centroids, axis=1)) for i in range(data.shape[0])])

    return labels, centroids

# Ejecutar el algoritmo MacQueen k-means
labels, centroids = macqueen_kmeans(scaledNonBinaryVariables, n_clusters=3)
print(labels)
print(centroids)

scaledNonBinaryVariables['cluster_MacQueen'] = labels

# Variables que deseas graficar (puedes cambiar 'Income' y 'MntWines' por las que prefieras)
x_var = 'Income'
y_var = 'MntWines'

# Crear el gráfico de dispersión con los clústeres
plt.figure(figsize=(10, 6))
plt.scatter(scaledNonBinaryVariables[x_var], scaledNonBinaryVariables[y_var], c=scaledNonBinaryVariables['cluster_MacQueen'], cmap='viridis', marker='o', edgecolor='k')

# Añadir etiquetas y título
plt.xlabel(x_var)
plt.ylabel(y_var)
plt.title(f'Distribución de los clústeres en {x_var} y {y_var}')
plt.colorbar(label='cluster_MacQueen')

# Mostrar el gráfico
plt.show()

"""### Elkan’s k-Means
El algoritmo de Elkan usa triangulaciones para reducir el número de cálculos de distancia y hacer el k-means más eficiente. sklearn tiene este algoritmo.
"""

def elkan_kmeans(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', algorithm='elkan')
    kmeans.fit(df)
    return kmeans.labels_, kmeans.cluster_centers_

labels_elkans, centroids = elkan_kmeans(scaledNonBinaryVariables, n_clusters=3)

scaledNonBinaryVariables['cluster_Elkans'] = labels_elkans

# Variables que deseas graficar (puedes cambiar 'Income' y 'MntWines' por las que prefieras)
x_var = 'Income'
y_var = 'MntWines'

# Crear el gráfico de dispersión con los clústeres
plt.figure(figsize=(10, 6))
plt.scatter(scaledNonBinaryVariables[x_var], scaledNonBinaryVariables[y_var], c=scaledNonBinaryVariables['cluster_Elkans'], cmap='viridis', marker='o', edgecolor='k')

# Añadir etiquetas y título
plt.xlabel(x_var)
plt.ylabel(y_var)
plt.title(f'Distribución de los clústeres en {x_var} y {y_var}')
plt.colorbar(label='cluster_Elkans')

# Mostrar el gráfico
plt.show()

"""### Soft k-Means (Fuzzy k-Means)
Este algoritmo difiere de k-means estándar porque no asigna cada punto de manera exclusiva a un clúster. En su lugar, cada punto tiene un grado de pertenencia a cada clúster.

Para esto, se puede usar la implementación de fuzzy-c-means en Python.
"""

#!pip install scikit-fuzzy
import numpy as np
import skfuzzy as fuzz

def soft_kmeans(df, n_clusters):
    df_array = np.array(df.T)  # Transponer el dataframe
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(df_array, n_clusters, 2, error=0.005, maxiter=1000)
    return u, cntr

# Grado de pertenencia a los clusters y centroides
memberships, centroids = soft_kmeans(scaledNonBinaryVariables, n_clusters=3)

# Obtener el índice del clúster con la mayor probabilidad para cada instancia
cluster_labels = np.argmax(memberships, axis=0)

# Agregar las etiquetas de clúster más probables al dataframe
scaledNonBinaryVariables['cluster_SoftKMeans'] = cluster_labels

# Ver las primeras filas del dataframe con las etiquetas de clúster asignadas
print(scaledNonBinaryVariables.head())

"""### Comparar clústeres generados por los diferentes algoritmos

* Silhouette
  - Valores cercanos a 1: Indican que los clústeres están bien separados y los puntos están correctamente asignados a sus clústeres.
  - Valores cercanos a 0: Indican que los clústeres se superponen o que los puntos están en los bordes de los clústeres.
  - Valores negativos: Indican que los puntos están mal agrupados, probablemente asignados al clúster incorrecto.
* Calinski-Harabasz
  - Un valor alto indica que los clústeres están bien separados y son compactos, lo cual es deseable en un buen agrupamiento.
  - Un valor bajo indica que los clústeres son más dispersos internamente o están superpuestos, peor calidad de agrupamiento.
* Davies-Bouldin
  - Valores más bajos indican un mejor clustering. Esto significa que los clústeres son más compactos y están bien separados entre sí.
  - Valores altos indican que los clústeres están más dispersos internamente o están demasiado cerca unos de otros, lo que sugiere un mal ajuste del modelo de clustering.
"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def calcular_metricas_clustering(df, columns):
    """
    Función para calcular métricas de evaluación de clustering para diferentes conjuntos de etiquetas.

    Parámetros:
    - df: DataFrame con las variables escaladas y las columnas de etiquetas.
    - columns: Lista de nombres de las columnas que contienen las etiquetas de los clusters.

    Retorna:
    - Un diccionario con los resultados de las métricas para cada conjunto de etiquetas.
    """
    resultados = {}

    for col in columns:
        labels = df[col]

        # Calculamos las métricas ignorando las columnas de los clusters
        X = df.drop(columns=columns)

        # Silhouette Score
        sil_score = silhouette_score(X, labels)

        # Calinski-Harabasz Score
        calinski_score = calinski_harabasz_score(X, labels)

        # Davies-Bouldin Index
        db_index = davies_bouldin_score(X, labels)

        # Guardamos los resultados en un diccionario
        resultados[col] = {
            'Silhouette Score': sil_score,
            'Calinski-Harabasz': calinski_score,
            'Davies-Bouldin': db_index
        }

    return resultados

# Llamar la función
columns = ['Cluster', 'cluster_MacQueen', 'cluster_Elkans', 'cluster_SoftKMeans']
resultados = calcular_metricas_clustering(scaledNonBinaryVariables, columns)

# Mostrar resultados
for col, metricas in resultados.items():
    print(f"{col} -> Silhouette Score: {metricas['Silhouette Score']}, "
          f"Calinski-Harabasz: {metricas['Calinski-Harabasz']}, "
          f"Davies-Bouldin: {metricas['Davies-Bouldin']}")

"""## Clustering aglomerativo

Principales parámetros de AgglomerativeClustering:

n_clusters:

Descripción: El número de clústeres que se desea obtener.
Valor por defecto: 2.
Tipo: Entero o None. Si es None, el algoritmo fusiona todos los puntos en un solo clúster.

affinity (eliminado en versiones recientes, sustituido por metric):

Descripción: Especifica la métrica utilizada para calcular la distancia entre los puntos de datos.
Valores posibles:
'euclidean' (predeterminado).
Otras métricas: 'manhattan', 'cosine'.
Tipo: Cadena (string).
Nota: Desde la versión 1.2 de scikit-learn, este parámetro ha sido reemplazado por metric.

metric:

Descripción: Especifica la métrica de distancia utilizada en el clustering. Reemplaza el parámetro affinity.
Valor por defecto: 'euclidean'.
Valores posibles: 'euclidean', 'manhattan', 'cosine'.

linkage:

Descripción: El criterio de enlace que determina la estrategia de combinación de clústeres.
Valores posibles:
'ward': Minimiza la varianza de los clusteres fusionados. Solo se puede usar con la distancia euclidiana.
'complete': Maximiza la distancia entre los puntos más lejanos de los clusters.
'average': Usa la media de las distancias de los puntos de los clusters.
'single': Usa la distancia mínima entre puntos en los clusters.

distance_threshold:

Descripción: Distancia para detener el proceso de aglomeración. Si se especifica este parámetro, el algoritmo forma clústeres hasta que la distancia entre ellos exceda este valor.
Valor por defecto: None. Si no se especifica, el número de clústeres es controlado por n_clusters.

"""

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Visualización del dendrograma para determinar el número de clusters
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(scaledNonBinaryVariables, method='ward'))
plt.title('Dendrograma')
plt.xlabel('Puntos de datos')
plt.ylabel('Distancia Euclidiana')
plt.show()

# Aplicación del algoritmo jerárquico (usando AgglomerativeClustering)
# Puedes ajustar el número de clusters (n_clusters) según el dendrograma
hierarchical_clustering = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
clusters = hierarchical_clustering.fit_predict(scaledNonBinaryVariables)

# Añadimos los clusters obtenidos a nuestro DataFrame original
scaledNonBinaryVariables['cluster_hierarchical'] = clusters

# Visualización rápida de los clusters en el DataFrame
print(scaledNonBinaryVariables.head())

columns = ['Cluster', 'cluster_MacQueen', 'cluster_Elkans', 'cluster_SoftKMeans', 'cluster_hierarchical']
resultados = calcular_metricas_clustering(scaledNonBinaryVariables, columns)

# Mostrar resultados
for col, metricas in resultados.items():
    print(f"{col} -> Silhouette Score: {metricas['Silhouette Score']}, "
          f"Calinski-Harabasz: {metricas['Calinski-Harabasz']}, "
          f"Davies-Bouldin: {metricas['Davies-Bouldin']}")

# Aplicación del algoritmo jerárquico (usando AgglomerativeClustering)
# Puedes ajustar el número de clusters (n_clusters) según el dendrograma
hierarchical_clustering1 = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
clusters = hierarchical_clustering1.fit_predict(scaledNonBinaryVariables)

# Añadimos los clusters obtenidos a nuestro DataFrame original
scaledNonBinaryVariables['cluster_hierarchical1'] = clusters

# Visualización rápida de los clusters en el DataFrame
print(scaledNonBinaryVariables.head())

columns = ['Cluster', 'cluster_MacQueen', 'cluster_Elkans', 'cluster_SoftKMeans', 'cluster_hierarchical', 'cluster_hierarchical1']
resultados = calcular_metricas_clustering(scaledNonBinaryVariables, columns)

# Mostrar resultados
for col, metricas in resultados.items():
    print(f"{col} -> Silhouette Score: {metricas['Silhouette Score']}, "
          f"Calinski-Harabasz: {metricas['Calinski-Harabasz']}, "
          f"Davies-Bouldin: {metricas['Davies-Bouldin']}")

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Visualización del dendrograma para determinar el número de clústeres
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(scaledNonBinaryVariables, method='complete'))
plt.title('Dendrograma')
plt.xlabel('Puntos de datos')
plt.ylabel('Distancia Euclidiana')
plt.show()

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Visualización del dendrograma para determinar el número de clústeres
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(scaledNonBinaryVariables, method='single'))
plt.title('Dendrograma')
plt.xlabel('Puntos de datos')
plt.ylabel('Distancia Euclidiana')
plt.show()

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Visualización del dendrograma para determinar el número de clusters
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(scaledNonBinaryVariables, method='ward'))
plt.title('Dendrograma')
plt.xlabel('Puntos de datos')
plt.ylabel('Distancia Euclidiana')
plt.show()