#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 13:41:08 2025
@author: noob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# STEP 1: Construct DataFrame
data = {
    'Y_actual': [
        0.1250, 1.8500, 0.7750, 1.4000, 0.8500, 0.3250, 0.9000, 0.0250, 1.6250, 0.7250,
        0.5750, 1.7000, 1.1500, 0.8000, 0.4750, 0.1500, 1.5500, 0.2500, 1.2000, 1.7750,
        1.2500, 1.7250, 1.0250, 0.2750, 0.7000, 1.8250, 1.4250, 0.8750, 0.3000, 1.8000,
        1.5250, 1.9250, 0.6000, 1.0000, 1.7500, 1.0750, 0.2250, 1.5000, 1.6750, 0.0750,
        1.3500, 0.4500, 0.0500, 1.3000, 1.1250, 1.1750, 1.3250, 0.1000, 1.4750, 0.6250,
        0.4250, 0.6500, 1.5750, 1.2750, 0.1750, 1.8750, 0.2000, 0.7500, 0.9250, 0.5500,
        0.3750, 2.0000, 0.3500, 0.4000, 1.0500, 1.4500, 1.1000, 1.9500, 1.3750, 1.2250,
        0.8250, 0.9750, 0.5000, 1.9000, 1.9750, 1.6000, 1.6500, 0.5250, 0.6750, 0.9500
    ],
    'Y_predicted': [
        0.0794, 1.8100, 0.7907, 1.4185, 0.8030, 0.3070, 0.9033, 0.0387, 1.6495, 0.7001,
        0.6076, 1.7544, 1.1902, 0.8013, 0.4787, 0.1169, 1.6065, 0.1929, 1.1863, 1.7854,
        1.2533, 1.6925, 1.0246, 0.2722, 0.6672, 1.8315, 1.2800, 0.9025, 0.2641, 1.8620,
        1.6227, 1.8301, 0.5647, 0.9134, 1.6549, 1.0973, 0.2316, 1.6160, 1.6982, 0.0554,
        1.3385, 0.4659, 0.0022, 1.3123, 1.2260, 1.3033, 1.3378, 0.1078, 1.6392, 0.6214,
        0.4224, 0.6835, 1.4065, 1.3368, 0.1707, 1.8861, 0.2128, 0.7538, 0.8515, 0.5682,
        0.3667, 1.9918, 0.3102, 0.3867, 1.0337, 1.5364, 1.1376, 1.8295, 1.4126, 1.3222,
        0.7916, 0.9057, 0.4832, 1.9002, 1.9773, 1.6875, 1.7291, 0.4889, 0.6581, 0.9418
    ]
}
df = pd.DataFrame(data)

# STEP 2: Train XGBoost model
X_input = df[['Y_predicted']]
y_output = df['Y_actual']

xgb_model = XGBRegressor(
    n_estimators=3000,
    max_depth=300,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
xgb_model.fit(X_input, y_output)

# STEP 3: Predictions for fit line
x_vals = np.linspace(df['Y_predicted'].min(), df['Y_predicted'].max(), 300).reshape(-1, 1)
y_range_pred = xgb_model.predict(x_vals)

# STEP 4: Plot fit
plt.figure(figsize=(8, 6))
plt.scatter(df['Y_predicted'], df['Y_actual'], alpha=0.6, label='Data')
plt.plot(x_vals, y_range_pred, color='green', label='XGBoost Fit')
plt.xlabel('Y_predicted')
plt.ylabel('Y_actual')
plt.title('XGBoost Fit: Y_actual vs Y_predicted')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 5: Residual Plot
y_fitted = xgb_model.predict(X_input)
residuals = y_output - y_fitted

plt.figure(figsize=(8, 5))
plt.scatter(df['Y_predicted'], residuals, color='darkorange', alpha=0.6)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Y_predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot (XGBoost)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: MSE
mse = mean_squared_error(y_output, y_fitted)
print(f"\nXGBoost Regression MSE: {mse:.6f}")



# STEP 7: Predict all values in dataset and print comparison
df['Y_predicted_by_model'] = xgb_model.predict(X_input)

print("\nActual vs Predicted by XGBoost Model:")
print(df[['Y_actual', 'Y_predicted_by_model']])



z = df[['Y_actual']].copy()
z['Y_predicted_by_model'] = xgb_model.predict(X_input)
z['error'] = np.abs(z['Y_actual'] - z['Y_predicted_by_model'])



import pickle

# Save the model as a pickle file
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

print("Model saved as xgb_model.pkl")
