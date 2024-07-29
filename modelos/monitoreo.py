import numpy as np

# Función para monitorear el modelo
def monitor_model(model, X_new, y_true):
    predictions = model.predict(X_new)
    mse = mean_squared_error(y_true, predictions)
    return mse

# Simulación de monitoreo periódico (por ejemplo, diariamente)
# En producción, este código estaría automatizado y ejecutado en intervalos regulares
X_new, y_new = X_test, y_test  # En la realidad, X_new e y_new serían datos nuevos

rf_mse_new = monitor_model(rf_model, X_new, y_new)
xgb_mse_new = monitor_model(xgb_model, X_new, y_new)

print(f'Nuevo Random Forest MSE: {rf_mse_new}')
print(f'Nuevo XGBoost MSE: {xgb_mse_new}')

# Verificar si los errores han aumentado significativamente
if rf_mse_new > rf_mse * 1.1:  # umbral del 10% de incremento
    print("Alerta: El error del modelo Random Forest ha aumentado significativamente.")
if xgb_mse_new > xgb_mse * 1.1:  # umbral del 10% de incremento
    print("Alerta: El error del modelo XGBoost ha aumentado significativamente.")
