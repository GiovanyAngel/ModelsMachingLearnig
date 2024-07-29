import pandas as pd
import matplotlib.pyplot as plt

# Primero realizamos la carga de los datos
df = pd.read_csv('depositos_oinks.csv')

# Realizamos una exploración inicial de los datos usando los metodos head e info
print(df.head())
print(df.info())

# Convertimos las columnas de fecha en tipo datetime
df['operation_date'] = pd.to_datetime(df['operation_date'])
df['user_createddate'] = pd.to_datetime(df['user_createddate'])

# Ahora definimos las métricas pertinentes
# 1. Definición de frecuencia de depósitos
df['deposit_count'] = df.groupby('user_id')['operation_value'].transform('count')

# Definición para el valor total de depósitos
df['total_deposit_value'] = df.groupby('user_id')['operation_value'].transform('sum')

# Definición de la antiguedad del ususario
df['user_age_days'] = (df['operation_date'] - df['user_createddate']).dt.days

# Definición de la métrica combinada (frecuencia * total_deposit_value) / antiguedad
df['user_score'] = (df['deposit_count'] * df['total_deposit_value']) / df['user_age_days']

# Ahora daremos una calificación a los usuarios
user_scores = df.groupby('user_id')['user_score'].mean().sort_values(ascending=False)

# Ahora visualizaremos los resultados
plt.figure(figsize=(10, 6))
user_scores.plot(kind='bar')
plt.title('User Scores')
plt.xlabel('User ID')
plt.ylabel('Score')
plt.show()
