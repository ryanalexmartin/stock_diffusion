import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split

generate_data = False


class LDM:
    def __init__(self, input_dim, hidden_dim):
        self.reverse_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))
        self.optim = Adam(self.reverse_model.parameters())

    def forward_process(self, X, beta_t=torch.Tensor([0.5])):
        noise = torch.empty(X.shape).normal_(mean=0, std=beta_t.sqrt().item())
        return (1 - beta_t).sqrt() * X + noise

    def reverse_process(self, Y):
        return self.reverse_model(Y)

    def fit(self, X, epochs=1000):
        X_tensor = torch.Tensor(X)
        for epoch in range(epochs):
            self.optim.zero_grad()
            Y = self.forward_process(X_tensor)
            pred = self.reverse_process(Y)
            loss = nn.MSELoss()(pred, X_tensor)
            loss.backward()
            self.optim.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss {loss.item()}')


if (generate_data):
    # Generate a more complex synthetic dataset
    T = 10000  # Increase the time points for more data
    mu = 0.001
    sigma = 0.01
    S0 = 100
    np.random.seed(0)
    dt = 1 / T
    t = np.linspace(0, 1, T)
    W = np.random.standard_normal(size=T)
    # The scale of the Brownian motion is increased
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5*sigma**2)*t + sigma*W
    S = S0 * np.exp(X)

    # Reshape for neural network (assume we predict one step ahead using
    # N previous steps)
    N = 10
    X = np.array([S[i-N:i] for i in range(N, len(S))])
    y = S[N:].reshape(-1, 1)

    # Create train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

# save the data to new files
np.savetxt("X_train.txt", X_train)
np.savetxt("X_test.txt", X_test)
np.savetxt("y_train.txt", y_train)
np.savetxt("y_test.txt", y_test)

# load the data from the files
X_train = np.loadtxt("X_train.txt")
X_test = np.loadtxt("X_test.txt")
y_train = np.loadtxt("y_train.txt")
y_test = np.loadtxt("y_test.txt")

# Initialize and train the model
ldm = LDM(N, 50)
ldm.fit(X_train, epochs=5000)
