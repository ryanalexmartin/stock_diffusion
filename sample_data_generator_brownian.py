import numpy as np
import matplotlib.pyplot as plt

# Set parameters for simulation
T = 100  # Time points (days)
mu = 0.001  # Expected return (drift)
sigma = 0.01  # Expected volatility
S0 = 50  # Initial stock price

# Simulation of Geometric Brownian Motion
# np.random.seed(0) # seed 0 for validating results
np.random.seed()
dt = 1 / T
t = np.linspace(0, 1, T)
W = np.random.standard_normal(size=T)
# Cumulative sum of standard normal random variables scaled by sqrt(dt)
W = np.cumsum(W)*np.sqrt(dt)
X = (mu-0.5*sigma**2)*t + sigma*W
S = S0*np.exp(X)

# Saving the simulated data
np.savetxt("data/stock_prices.txt", S)

# Plotting the simulated data and saving the plot to a file
plt.plot(S)
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Simulated Stock Prices')
plt.savefig('data/stock_prices.png')
plt.close()
