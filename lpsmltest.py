import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def computemetrics(model, X_test, Y_test):
    real_values = Y_test
    print("Accuracy: ", model.score(X_test, Y_test))

    y_true = np.array(real_values)
    pred_values = model.predict(X_test)
    y_pred = np.array(pred_values)

    mae = mean_absolute_error(y_true, y_pred)
    print(f"Mean Absolute Error: {mae}")

    analysis_df = pd.DataFrame({
        'Real': y_true,
        'Previsto': y_pred,
        'Error_Absoluto': np.abs(y_true - y_pred),
        'Residual': y_true - y_pred
    })

    print(analysis_df)

    mape_decimal = mean_absolute_percentage_error(y_true, y_pred)

    mape_percentage = mape_decimal * 100

    print(f"MAPE: {mape_percentage:.2f}%")

    plt.figure(1, figsize=(6, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Fit')
    plt.xlabel('Valores Actual')
    plt.ylabel('Valores Previstos')
    plt.title(f'Actual vs Previsto (MAE: {mae:.2f})')
    plt.legend()
    plt.savefig("MAE.png")

    plt.figure(2, figsize=(10, 5))
    plt.hist(np.abs(y_true - y_pred), bins=30, color="steelblue",edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--", label="Perfect prediction")
    plt.title("Frequency of Error vs Magnitude of Error")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("Hist.png")
