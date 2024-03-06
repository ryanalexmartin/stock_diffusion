import numpy as np
from sklearn.model_selection import train_test_split
from scrapers.bitcoin_price import get_price_history
import matplotlib.pyplot as plt
from ldm.ldm import LatentDiffusionModel as LDM
from sklearn.metrics import mean_squared_error
import torch
import pandas as pd

df = get_price_history()
# Adapt the Bitcoin data to match the existing synthetic data format
S = df['price'].values  # Take only the price column

# Reshape for neural network (assume we predict one step ahead using N previous
# steps)
N = 100  # Number of previous steps to use for prediction
X = np.array([S[i-N:i] for i in range(N, len(S))])
y = S[N:].reshape(-1, 1)

# Assuming 'df' has a DateTimeIndex or a date column named 'time'
dates = df.index if isinstance(df.index, pd.DatetimeIndex) else df['time']

# Adjust dates array to align with y
dates = dates[N:]

# Create train/test sets
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, dates, test_size=0.2, shuffle=False
)


# visualize the data for train and test, to see if the split is correct
plt.figure(figsize=(30, 7))
plt.plot(dates_train, y_train, label='Train data', color='blue')
plt.plot(dates_test, y_test, label='Test data', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Train and Test Data')
plt.legend()
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.tight_layout()  # Adjust layout to make room for the rotated date labels
plt.savefig('bitcoin_price_train_test_data.png')


print("The model is training on dates from", dates_train[0], "to", dates_train[-1])
print("The model is testing on dates from", dates_test[0], "to", dates_test[-1])

# Initialize and train the model
ldm = LDM(N, 500)
# N is the number of previous steps to use for prediction, 50 is the number of
# neurons in the hidden layer

# Train the model, printing the loss every 100 epochs
ldm.fit(X_train, epochs=10000)

predictions = ldm.predict(X_train).flatten()
actuals = y_test.flatten()[:len(predictions)]

# Ensure same number of samples
predictions = predictions[:len(actuals)]

mse = mean_squared_error(actuals, predictions)
print(f"Test MSE: {mse}")


# Print the predictions and actuals
print("Predictions:")
print(predictions)
print("Actual data:")
print(actuals)

# Calculate mean squared error (MSE)
mse = mean_squared_error(actuals, predictions)
print(f"Test MSE: {mse}")

full_prices_df = pd.DataFrame({
    'Date': dates,
    'Actual Prices': S[N:]
})

full_predictions_df = pd.DataFrame({
    'Date': dates_test,
    'Predicted Prices': predictions.flatten(),
})

plt.figure(figsize=(30, 7))
plt.plot(full_prices_df['Date'], full_prices_df['Actual Prices'], label='Actual Prices', color='blue')
plt.plot(full_predictions_df['Date'], full_predictions_df['Predicted Prices'], label='Predicted Prices', linestyle='--', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Predictions vs Actuals')
plt.legend()
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.tight_layout()  # Adjust layout to make room for the rotated date labels

plt.savefig('bitcoin_price_predictions.png')

