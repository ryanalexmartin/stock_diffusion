# Description: A simple experiment to demonstrate the use of a simple latent
# diffusion model (LDM) for time series data.
import numpy as np
import matplotlib
from sklearn.linear_model import LinearRegression
matplotlib.use('Agg')  # Use a non-interactive backend

# Loading the data
data = np.loadtxt("data/stock_prices.txt")

# Reshaping data (if necessary) and preparing train/test sets
data = data.reshape(-1, 1)
train_data = data[:-10]  # Using all but the last 10 data points for training
test_data = data[-10:]  # Reserving last 10 data points for testing


class SimpleLDM:
    def __init__(self):
        self.reverse_model = LinearRegression()

    def forward_process(self, X, beta_t):
        noise = np.random.normal(0, beta_t, size=X.shape)
        return np.sqrt(1 - beta_t) * X + noise

    def reverse_process(self, Y, beta_t):
        return self.reverse_model.predict(Y)

    def fit(self, X):
        beta_t = 0.1  # Assume a fixed beta_t for simplicity
        Y = self.forward_process(X, beta_t)
        self.reverse_model.fit(Y, X)

    def predict(self, Y):
        return self.reverse_model.predict(Y)


# Creating and training the model
ldm = SimpleLDM()
ldm.fit(train_data)  # Training on our train dataset

# Predicting the last few data points and comparing to actual data
predictions = ldm.predict(test_data)
print("Predictions:")
print(predictions)
print("Actual data:")
print(test_data)

# Calculate Mean Squared Error Error as a basic metric for this testing
mse = np.mean((predictions - test_data) ** 2)
print(f"Mean Squared Error: {mse}")
