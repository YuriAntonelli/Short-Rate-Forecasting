# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:05:07 2023

@author: yuria
"""

# simulate CIR model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.special import iv
from scipy.optimize import minimize
#------------------------------------------------------------------------------

# simulate interest rate data usng the CIR model
def simulate_CIR(k, theta, sigma, r0, T, N):
    dt = T / N
    time = np.linspace(0, T, N+1)
    interest_rate_paths = np.zeros(N+1)
    interest_rate_paths[0] = r0
    
    for t in range(1, N+1):
        Z = np.random.randn()
        r = interest_rate_paths[t-1]
        interest_rate_paths[t] = r + k * (theta-r) * dt + sigma * np.sqrt(dt) * np.sqrt(max(0, r)) * Z
        
    return pd.Series(interest_rate_paths, index = time, name='Interest_Rate')

# Simulate CIR data
k_true = 2  # True mean reversion speed
theta_true = 0.1  # True long-term mean
sigma_true = 0.5  # True volatility of interest rates
r0_true = 0.03  # True initial interest rate
T = 1  # Time horizon
N = 10  # Number of time steps
dt = T/N
simulated_data = simulate_CIR(k_true, theta_true, sigma_true, r0_true, T, N)

# Create a DataFrame with lagged and current interest rates
lagged_interest_rates = simulated_data.shift(1).rename('Rate_t-1')
current_interest_rates = simulated_data.rename('Rate_t')
df = pd.concat([current_interest_rates, lagged_interest_rates], axis=1).dropna()

# Linear regression to estimate CIR parameters
X = df['Rate_t-1']
y = df['Rate_t']
model = LinearRegression()

# feature engeneering to fit the theoretical model
y = (y - X) / np.sqrt(X)
z1 = dt / np.sqrt(X)
z2 = dt * np.sqrt(X)
X = np.column_stack((z1, z2))

# Build the model
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

# Calculate the predicted values (y_hat), residuals and the parameters
y_hat = model.predict(X)
residuals = y - y_hat
beta1 = model.coef_[0]        
beta2 = model.coef_[1]

# get the parameter of interest for CIR
k_estimate = -beta2
theta_estimate = beta1/k_estimate
sigma_estimate = np.std(residuals)/np.sqrt(dt)

# Print the estimated Vasicek parameters
print('Estimated CIR parameters:')
print(f'Mean Reversion Speed (k): {k_estimate:.6f} instead of {k_true}')
print(f'Long-term Mean (theta): {theta_estimate:.6f} instead of {theta_true}')
print(f'Volatility (sigma): {sigma_estimate:.6f} instead of {sigma_true}')

######################

"""
# Plot the simulated interest rate paths
plt.figure(figsize=(15, 6))
plt.plot(simulated_data.index, simulated_data.values, label='Simulated Data')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.title('Simulated CIR Model Interest Rate Paths')
plt.grid(True)
plt.legend()
plt.show()
"""

###############################################################################

# MAXIMUM LIKELIHOOD CALIBRATION
# Initialize stuff
X = df[['Rate_t-1']]
y = df['Rate_t']
c = 0
k = 0

def u(r_s):
    return c*r_s*np.exp(-k*dt)

def v(r_t):
    return c*r_t

def CIR_log_likelihood(params, y, X):
    k = params[0] 
    theta = params[1]
    sigma = params[2]
    log_likelihood = 0
    
    for n in range(len(y)):
        r_s = X.iloc[n,:]
        r_t = y.iloc[n]
        
        c = 2*k / ((sigma**2)*(1-np.exp(-k*dt)))
        q = (2*k*theta / (sigma**2)) - 1
        I = iv(2.5, 2*np.sqrt(v(r_t)*u(r_s)))
        coeff = np.exp(-2*np.sqrt(v(r_t)*u(r_s)))
        I_rescaled = I * coeff
        
        log_likelihood += -u(r_t)-v(r_s)*np.log(c)+(q/2)*np.log(v(r_s)/u(r_t))+np.log(I_rescaled[0])
        log_likelihood = log_likelihood[0]
    return -log_likelihood
        
initial_params = (k_estimate, theta_estimate, sigma_estimate)
result = CIR_log_likelihood(initial_params, y, X)
print(f'the log_likelihood is: {result}')


# TEST DELLA FUNZIONE
k = k_estimate
theta = theta_estimate
sigma = sigma_estimate
log_likelihood = 0
for n in range(len(y)):
    r_s = X.iloc[n,:]
    r_t = y.iloc[n]
    
    c = 2*k / ((sigma**2)*(1-np.exp(-k*dt)))
    q = (2*k*theta / (sigma**2)) - 1
    I = iv(2.5, 2*np.sqrt(v(r_t)*u(r_s)))
    coeff = np.exp(-2*np.sqrt(v(r_t)*u(r_s)))
    I_rescaled = I * coeff
    
    log_likelihood += -u(r_t)-v(r_s)*np.log(c)+(q/2)*np.log(v(r_s)/u(r_t))+np.log(I_rescaled[0])
    log_likelihood = log_likelihood[0]
print(f'the log_likelihood is: {log_likelihood}')


"""
# ML OPTIMIZATION
ml_estimated_params = minimize(CIR_log_likelihood, initial_params, args=(y, X))

# Extract ML parameter estimates
k_ml = ml_estimated_params.x[0]
theta_ml = ml_estimated_params.x[1]
sigma_ml = ml_estimated_params.x[2]

# Print the ML estimated Vasicek parameters
print('-------------------------------------------')
print('Estimated CIR parameters using MLE:')
print(f'Mean Reversion Speed (k): {k_ml:.6f}')
print(f'Long-term Mean (theta): {theta_ml:.6f}')
print(f'Volatility (sigma): {sigma_ml:.6f}')
"""

"""
# Plot
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
        params = (k_v[i], theta_v[j], sigma_estimate) # to change sigma
        log_lik[i, j] = CIR_log_likelihood(params, y, X)

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
loglike_optimal = CIR_log_likelihood((k_optimal, theta_optimal, sigma_ml), y, X)
ax.scatter([k_optimal], [theta_optimal], [loglike_optimal], c='red', marker='o', s=200, zorder=2)

plt.show()
"""
































