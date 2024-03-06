import numpy as np
from sklearn.model_selection import train_test_split
from scrapers.bitcoin_price import get_price_history
from ldm.ldm import LatentDiffusionModel as LDM
from sklearn.metrics import mean_squared_error
import torch

df = get_price_history()
# Adapt the Bitcoin data to match the existing synthetic data format
S = df['price'].values  # Take only the price column

# Reshape for neural network (assume we predict one step ahead using N previous
# steps)
N = 10  # Number of previous steps to use for prediction
X = np.array([S[i-N:i] for i in range(N, len(S))])
y = S[N:].reshape(-1, 1)

# Create train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Initialize and train the model
ldm = LDM(N, 50)
# N is the number of previous steps to use for prediction, 50 is the number of
# neurons in the hidden layer

# Train the model, printing the loss every 100 epochs
ldm.fit(X_train, epochs=50000)


def make_predictions(model, test_input):
    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        test_input_tensor = torch.tensor(test_input, dtype=torch.float)
        predictions = model.reverse_model(test_input_tensor)
    # Ensuring predictions are flattened if necessary
    predictions = predictions.view(-1).numpy()
    return predictions


# Make predictions
# Assuming your test data is stored in X_test
predictions = make_predictions(ldm, X_test)

# Assuming y_test is a numpy array. If it's not, you might need to adjust
# accordingly. Adjust in case there's a shape mismatch
actuals = y_test[:predictions.shape[0]]
# Making sure we're matching the shapes
actuals = actuals.flatten()  # This makes sure actuals is a 1-D array

print("Shape of actuals:", actuals.shape)
print("Shape of predictions:", predictions.shape)

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
