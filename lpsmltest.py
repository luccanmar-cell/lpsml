import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def computemetrics(model, X_test, Y_test, X_train):
    real_values = Y_test
    print("Accuracy: ", model.score(X_test, Y_test))
    
    # Read the full original dataset
    df = pd.read_excel("newtarifa.xlsx", engine="openpyxl")
    
    y_true = np.array(real_values)
    pred_values = model.predict(X_test)
    y_pred = np.array(pred_values)
    
    # Calculate percentage error
    percentage_error = ((y_true - y_pred) / y_true) * 100
    
    # Initialize new columns with NaN for training rows (signifies not predicted)
    df["Prima Previsto"] = np.nan
    df["Porcentaje Error"] = np.nan
    
    # Get the indices of test samples (Y_test has the original indices)
    test_indices = Y_test.index.tolist()
    
    # Fill in predictions and errors only for test rows
    for idx, pred, error in zip(test_indices, pred_values, percentage_error):
        df.loc[idx, "Prima Previsto"] = float(pred)
        df.loc[idx, "Porcentaje Error"] = float(error)

    mae = mean_absolute_error(y_true, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # Create analysis dataframe for visualizations
    analysis_df = pd.DataFrame({
        'Real': y_true,
        'Previsto': y_pred,
        'Porcentaje Error': percentage_error,
        'Error_Absoluto': np.abs(y_true - y_pred),
        'Residual': y_true - y_pred
    })

    print(analysis_df)

    mape_decimal = mean_absolute_percentage_error(y_true, y_pred)

    mape_percentage = mape_decimal * 100

    print(f"MAPE: {mape_percentage:.2f}%")

    plt.figure(1, figsize=(6, 6))
    avg = np.mean(y_true)
    y_true_norm = y_true / avg
    y_pred_norm = y_pred / avg
    plt.scatter(y_true_norm, y_pred_norm, color='blue', alpha=0.7, label='Prediccion')
    plt.plot([y_true_norm.min(), y_true_norm.max()], [y_true_norm.min(), y_true_norm.max()], 'r--', label='Perfect Fit')
    plt.xlabel('Valores Actual / Promedio de y_true')
    plt.ylabel('Valores Previstos / Promedio de y_true')
    plt.title(f'Actual vs Previsto (MAE: {mae:.2f}, MAPE: {mape_percentage:.2f}%)')
    plt.legend()
    plt.savefig("MAE.png")

    plt.figure(2, figsize=(10, 5))
    plt.hist(np.abs(y_true - y_pred), bins=30, color="steelblue", edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--", label="Prediccion Perfecta")
    plt.title("Frecuencia de Error vs Magnitud de Error")
    plt.xlabel("Magnitud")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.savefig("Hist_error.png")
    
    # Save full dataframe with predictions and errors to Excel
    df.to_excel("newtarifa.xlsx", index=False)
