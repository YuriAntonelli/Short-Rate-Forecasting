# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 22:08:31 2023

@author: yuria
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os 

#######

# Change the working directory to a new path
new_directory = "C:/Users/yuria/Desktop/tesi/Codice"
os.chdir(new_directory)

########

df = pd.read_excel('EURIBOR.xlsx')
df.head()
dates = df.Date
euribor_ir = df.iloc[:, 1] / 100
df = euribor_ir.copy()
df = pd.DataFrame({'series1': df})
df = df.join(euribor_ir.shift(1),  how='left')
df.columns = ['rates t', 'rates t-1']
df = df.dropna()
df.reset_index(inplace=True)
df = df.iloc[:, 1:]


# for Vasicek
y = df['rates t']
X = df[['rates t-1']]

# for CIR
shift = np.percentile(y, 99)
y_shift = y.copy() + shift
X_shift = X.copy() + shift
X_shift = X_shift['rates t-1']

#----------------------------------------------------------------------

# Vasicek Calibration
model = LinearRegression()
model.fit(X, y)

# Extract parameter estimates
slope = model.coef_[0]
intercept = model.intercept_
dt = 1/252
k_estimate = (1-slope)/dt
theta_estimate = intercept / (1 - slope)
sigma_estimate = np.std(y - model.predict(X)) / np.sqrt(dt)

# Print the estimated Vasicek parameters
print('Estimated Vasicek parameters:')
print(f'Mean Reversion Speed (k): {k_estimate}')
print(f'Long-term Mean (theta): {theta_estimate}')
print(f'Volatility (sigma): {sigma_estimate}')

# Prediction
r0 = y.iloc[-1]

pred1 = r0 * np.exp(-k_estimate*dt) + theta_estimate * (1 - np.exp(-k_estimate*dt)) 
pred7 = r0 * np.exp(-k_estimate*7*dt) + theta_estimate * (1 - np.exp(-k_estimate*7*dt)) 
pred30 = r0 * np.exp(-k_estimate*30*dt) + theta_estimate * (1 - np.exp(-k_estimate*30*dt)) 
pred90 = r0 * np.exp(-k_estimate*90*dt) + theta_estimate * (1 - np.exp(-k_estimate*90*dt)) 
print(f'the Vasicek predictions are {pred1}{pred7}{pred30}{pred90}')

#------------------------------------------------------------------------
print('------------------------------------------------')
#-----------------------------------------------------------------------

# CIR Calibration
y_cir = (y_shift - X_shift) / np.sqrt(X_shift)
z1 = dt / np.sqrt(X_shift)
z2 = dt * np.sqrt(X_shift)
X_cir = np.column_stack((z1, z2))

# Build the model
model = LinearRegression(fit_intercept=False)
model.fit(X_cir, y_cir)

# Calculate the predicted values (y_hat), residuals and the parameters
y_hat = model.predict(X_cir)
residuals = y_shift - y_hat
beta1 = model.coef_[0]        
beta2 = model.coef_[1]

# get the parameter of interest for CIR
k_estimate = -beta2
theta_estimate = beta1/k_estimate
sigma_estimate = np.std(residuals)/np.sqrt(dt)

# Print the estimated Vasicek parameters
print('Estimated CIR parameters:')
print(f'Mean Reversion Speed (k): {k_estimate}')
print(f'Long-term Mean (theta): {theta_estimate}')
print(f'Volatility (sigma): {sigma_estimate}')

# Prediction
r0 = y_shift.iloc[-1]

pred1 = r0 * np.exp(-k_estimate*dt) + theta_estimate * (1 - np.exp(-k_estimate*dt)) - shift
pred7 = r0 * np.exp(-k_estimate*7*dt) + theta_estimate * (1 - np.exp(-k_estimate*7*dt)) - shift
pred30 = r0 * np.exp(-k_estimate*30*dt) + theta_estimate * (1 - np.exp(-k_estimate*30*dt)) - shift
pred90 = r0 * np.exp(-k_estimate*90*dt) + theta_estimate * (1 - np.exp(-k_estimate*90*dt)) - shift
print(f'the CIR predictions are {pred1}{pred7}{pred30}{pred90}')