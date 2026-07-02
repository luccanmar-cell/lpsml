import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Sample data: True target vs Model predictions
y_true = np.array([250000, 300000, 150000, 200000, 400000])
y_pred = np.array([240000, 315000, 140000, 185000, 410000])

# Calculate MAE
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error: {mae}")  # Output: 12000.0

# Create an error analysis DataFrame
analysis_df = pd.DataFrame({
    'Actual': y_true,
    'Predicted': y_pred,
    'Absolute_Error': np.abs(y_true - y_pred),
    'Residual': y_true - y_pred
})

print(analysis_df)

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Predictions')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs Predicted (MAE: {mae:.2f})')
plt.legend()
plt.show()
