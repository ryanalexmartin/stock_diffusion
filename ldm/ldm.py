import torch
import torch.nn as nn
from torch.optim import Adam


class LatentDiffusionModel:
    def __init__(self, input_dim, hidden_dim):
        self.reverse_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output dimension is 1, as we're predicting a single value
        )
        self.optimizer = Adam(self.reverse_model.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()
        self.training = True  # Add a flag to keep track of the mode

    def forward_process(self, past_prices_tensor, beta_t=torch.Tensor([0.5])):
        noise = torch.empty(
            past_prices_tensor.shape
        ).normal_(mean=0, std=beta_t.sqrt().item())
        return (1 - beta_t).sqrt() * past_prices_tensor + noise

    def fit(self, past_prices, epochs=10000):
        self.train()  # Set the model to training mode
        past_prices_tensor = torch.tensor(past_prices, dtype=torch.float)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            current_prices_diffused = self.forward_process(past_prices_tensor)
            current_prices_reconstructed = self.reverse_model(
                current_prices_diffused)
            loss = self.loss_func(
                current_prices_reconstructed, past_prices_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        print("Training complete. Final loss =", loss.item())

    def predict(self, past_prices):
        self.training = False  # Disable training mode
        past_prices_tensor = torch.tensor(past_prices, dtype=torch.float)
        with torch.no_grad(): # Context-manager that disabled gradient calculation
            past_prices_diffused = self.forward_process(past_prices_tensor)
            predictions = self.reverse_model(past_prices_diffused)
        return predictions.numpy()  # Convert the predictions from PyTorch tensor to a numpy array

    def eval(self, past_prices, future_prices):
        self.training = False  # Disable training mode

        # Use the predict method to get predictions
        predictions = self.predict(past_prices)

        # Convert validation data and predictions to tensors for loss calculation
        future_prices_tensor = torch.tensor(future_prices, dtype=torch.float)
        predictions_tensor = torch.tensor(predictions, dtype=torch.float)

        # Compute the loss
        loss = self.loss_func(predictions_tensor, future_prices_tensor)

        print(f"Validation loss = {loss.item()}")

        self.training = True  # Enable training mode back

    def train(self):
        self.training = True  # Enable training mode


"""
An explanation of "X" and "Y" in the forward and reverse processes:

The forward process is a simple linear transformation of the input data X, with
some added noise. The noise is sampled from a normal distribution with a mean
of 0 and a standard deviation of beta_t. The forward process is defined as:

    Y = sqrt(1 - beta_t) * X + noise

So "X" is a sliding window of the input data, and "Y" is the transformed data
with added noise. The reverse process is a simple linear transformation of the
noisy data Y, which attempts to predict the original data X. The reverse
process is defined as:

    X_hat = reverse_model(Y)

So "Y" is the noisy data, and "X_hat" is the model's prediction of the original
data. The model is trained to minimize the mean squared error between its
predictions and the original data.

"""
