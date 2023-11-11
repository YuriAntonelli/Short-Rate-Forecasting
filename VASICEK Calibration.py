# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:35:08 2023

@author: yuria
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Simulate interest rate data using the Vasicek model
def simulate_vasicek(k, theta, sigma, r0, T, N):
    dt = T / N
    time = np.linspace(0, T, N+1)
    interest_rate_paths = np.zeros(N+1)
    interest_rate_paths[0] = r0

    for t in range(1, N+1):
        Z = np.random.randn()
        r = interest_rate_paths[t-1]
        interest_rate_paths[t] = r + k * (theta - r) * dt + sigma * np.sqrt(dt) * Z

    return pd.Series(interest_rate_paths, index=time, name='Interest_Rate')

# Simulate Vasicek interest rate data
k_true = 2  # True mean reversion speed
theta_true = 0.05  # True long-term mean
sigma_true = 0.02  # True volatility of interest rates
r0_true = 0.03  # True initial interest rate
T = 1  # Time horizon
N = 100  # Number of time steps
simulated_data = simulate_vasicek(k_true, theta_true, sigma_true, r0_true, T, N)

# Create a DataFrame with lagged and current interest rates
lagged_interest_rates = simulated_data.shift(1).rename('Rate_t-1')
current_interest_rates = simulated_data.rename('Rate_t')
df = pd.concat([current_interest_rates, lagged_interest_rates], axis=1).dropna()

# Linear regression to estimate Vasicek parameters
X = df[['Rate_t-1']]
y = df['Rate_t']
model = LinearRegression()
model.fit(X, y)

# Extract parameter estimates
slope = model.coef_[0]
intercept = model.intercept_
dt = T / N
k_estimate = (1-slope)/dt
theta_estimate = intercept / (1 - slope)
sigma_estimate = np.std(y - model.predict(X)) / np.sqrt(dt)

# Print the estimated Vasicek parameters
print('Estimated Vasicek parameters using LSE:')
print(f'Mean Reversion Speed (k): {k_estimate:.6f}')
print(f'Long-term Mean (theta): {theta_estimate:.6f}')
print(f'Volatility (sigma): {sigma_estimate:.6f}')

###################################################################################
"""
# Plot the simulated interest rate paths
plt.figure(figsize=(15, 6))
plt.plot(simulated_data.index, simulated_data.values, label='Simulated Data')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.title('Simulated Vasicek Model Interest Rate Paths')
plt.grid(True)
plt.legend()
plt.show()
"""
####################################################################################

# ML estimations
def vasicek_log_likelihood(params, y, X):
    k, theta, sigma = params
    likelihood = 0
    for n in range(len(y)):
        a = np.log(1/(np.sqrt(2*np.pi*sigma*dt)))
        b = (y.iloc[n] - (X.iloc[n,:]+k*(theta-X.iloc[n,:])*dt))**2
        c = 2*(sigma**2)*dt
        likelihood += a - (b/c)
        
    return -likelihood

initial_params = (k_estimate, theta_estimate, sigma_estimate)

ml_estimated_params = minimize(vasicek_log_likelihood, initial_params, args=(y, X))

# Extract ML parameter estimates
k_ml = ml_estimated_params.x[0]
theta_ml = ml_estimated_params.x[1]
sigma_ml = ml_estimated_params.x[2]

# Print the ML estimated Vasicek parameters
print('-------------------------------------------')
print('Estimated Vasicek parameters using MLE:')
print(f'Mean Reversion Speed (k): {k_ml:.6f}')
print(f'Long-term Mean (theta): {theta_ml:.6f}')
print(f'Volatility (sigma): {sigma_ml:.6f}')

#################################################################################
# plot the log_likelihood
array_length = 10
k_v = np.linspace(1, 10, array_length)
theta_v = np.linspace(0.03, 0.07, array_length)

# Create a meshgrid for 3D plotting
# Create grids for all possible combinations
grid1, grid2 = np.meshgrid(k_v, theta_v)

# Create an empty grid for log-likelihood values
log_lik = np.zeros((array_length, array_length))

# Calculate log-likelihood for all combinations of k and theta
for i in range(array_length):
    for j in range(array_length):
        params = (k_v[i], theta_v[j], sigma_ml)
        log_lik[i, j] = vasicek_log_likelihood(params, y, X)

# Create a 3D surface plot of the log-likelihood
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')
K, Theta = np.meshgrid(k_v, theta_v)

surf = ax.plot_surface(K, Theta, log_lik, cmap='viridis', rstride=1, cstride=1, alpha=0.8)

# Customize the plot
ax.set_xlabel('K')
ax.set_ylabel('Theta')
ax.set_zlabel('Log-Likelihood')
plt.title('Log-likelihood with fixed sigma', fontsize = 24)

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

# Add a red point at a specific coordinate (x_optimal, y_optimal)
k_optimal = k_ml  # Replace with your desired x coordinate
theta_optimal = theta_ml   # Replace with your desired y coordinate
loglike_optimal = vasicek_log_likelihood((k_optimal, theta_optimal, sigma_ml), y, X)
ax.scatter([k_optimal], [theta_optimal], [loglike_optimal], c='red', marker='o', s=200, zorder=2)

plt.show()




