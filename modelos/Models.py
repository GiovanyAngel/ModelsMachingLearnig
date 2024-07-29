import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Comenzamos con la carga de los datos
df = pd.read_excel('info_satisfaccion_trabajo.xlsx')

# Preprocesamiento de los datos
df = pd.get_dummies(df, drop_first=True)  # Convertir variables categóricas a dummy variables
X = df.drop(columns=['JobSatisfaction'])
y = df['JobSatisfaction']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelos
# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# XGBoost
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# Evaluar el desempeño de los modelos
rf_mse = mean_squared_error(y_test, rf_predictions)
xgb_mse = mean_squared_error(y_test, xgb_predictions)

print(f'Random Forest MSE: {rf_mse}')
print(f'XGBoost MSE: {xgb_mse}')
