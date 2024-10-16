# Instalar la biblioteca PuLP
!pip install pulp

# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus

# Datos de computadoras (puedes agregar más)
data = {
    'Modelo': ['PC1', 'PC2', 'PC3', 'PC4'],
    'Precio': [[1400000, 1500000, 1600000], [2400000, 2500000, 2600000], [1900000, 2000000, 2100000], [1700000, 1800000, 1900000]],
    'Memoria_RAM': [[7, 8, 9], [15, 16, 17], [7, 8, 9], [30, 32, 34]],  # en GB
    'Sistema_Operativo': ['Windows', 'Linux', 'Linux', 'Windows'],
    'Procesador': ['i5', 'i7', 'i5', 'i9'],
    'Placa_Video': ['GTX 1650', 'RTX 2060', 'GTX 1660', 'RTX 3070'],
    'Cuotas_Sin_Interes': [[5, 6, 7], [11, 12, 13], [5, 6, 7], [11, 12, 13]]
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Filtrar por restricción presupuestaria
presupuesto_min = 1000000
presupuesto_max = 2500000

# Calcular el valor medio de cada criterio
df['Precio'] = df['Precio'].apply(lambda x: np.mean(x))
df['Memoria_RAM'] = df['Memoria_RAM'].apply(lambda x: np.mean(x))
df['Cuotas_Sin_Interes'] = df['Cuotas_Sin_Interes'].apply(lambda x: np.mean(x))

# Filtrar computadoras dentro del presupuesto
df = df[(df['Precio'] >= presupuesto_min) & (df['Precio'] <= presupuesto_max)]

# Normalización de los datos (sin considerar precio en TOPSIS)
features = df[['Memoria_RAM', 'Cuotas_Sin_Interes']]  # Sin precio
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(features)

# Cálculo de las puntuaciones Fuzzy TOPSIS (sin precio)
weights = np.array([0.5, 0.5])  # Ajustar según la importancia de cada criterio
weighted_data = normalized_data * weights

# Alternativas ideales y anti-ideales
ideal_solution = np.max(weighted_data, axis=0)
anti_ideal_solution = np.min(weighted_data, axis=0)

# Distancias y puntajes
distance_to_ideal = np.sqrt(np.sum((weighted_data - ideal_solution) ** 2, axis=1))
distance_to_anti_ideal = np.sqrt(np.sum((weighted_data - anti_ideal_solution) ** 2, axis=1))
scores = distance_to_anti_ideal / (distance_to_ideal + distance_to_anti_ideal)

# Agregar scores al DataFrame
df['Score'] = scores

# Imprimir las puntuaciones calculadas
print("\nPuntuaciones Fuzzy TOPSIS:")
print(df[['Modelo', 'Score', 'Precio']])

# Optimización con Programación Lineal
model = LpProblem("Mejor_Alternativa_Computadora", LpMinimize)
x = LpVariable.dicts("x", df.index, lowBound=0, upBound=1, cat='Binary')

# Nueva función objetivo: maximizar puntaje TOPSIS mientras minimiza el precio
model += lpSum(df['Score'][i] * x[i] for i in df.index) - lpSum(df['Precio'][i] * x[i] for i in df.index)

# Restricción de presupuesto
model += lpSum(df['Precio'][i] * x[i] for i in df.index) <= presupuesto_max

# Asegurarse de seleccionar al menos una computadora
model += lpSum(x[i] for i in df.index) >= 1

# Resolver el modelo
model.solve()

# Mostrar resultados
print("Estado de la solución:", LpStatus[model.status])
print("\nValores de las variables de decisión:")
for i in df.index:
    print(f"{df.loc[i]['Modelo']}: {x[i].varValue}")

# Mostrar computadoras seleccionadas
print("\nComputadoras seleccionadas:")
for i in df.index:
    if x[i].varValue == 1:
        print(df.loc[i])

# Resultados ordenados por score
df = df.sort_values(by='Score', ascending=False)
print("\nRanking de computadoras según Fuzzy TOPSIS:")
print(df[['Modelo', 'Precio', 'Memoria_RAM', 'Sistema_Operativo', 'Procesador', 'Placa_Video', 'Cuotas_Sin_Interes', 'Score']])
