import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Set parameters for simulation
T = 100  # Time points (Days)
mu = 0.001  # Expected return (drift)
sigma = 0.01  # Expected volatility
S0 = 50  # Initial stock price

# Simulation of Geometric Brownian Motion
np.random.seed(0)
dt = 1 / T
t = np.linspace(0, 1, T)
W = np.random.standard_normal(size=T)
W = np.cumsum(W)*np.sqrt(dt)  # Cumulative sum of standard normal random variables scaled by sqrt(dt)
X = (mu-0.5*sigma**2)*t + sigma*W
S = S0*np.exp(X)  # Geometric Brownian Motion

# Plot the generated stock price data
plt.plot(S)
plt.ylabel('Stock Price')
plt.xlabel('Time (Days)')
plt.title('Generated Stock Price Data - Geometric Brownian Motion')
plt.savefig('stock_price_simulation.png')  # Save the plot to a file

