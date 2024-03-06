from sklearn.linear_model import LinearRegression

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


# Fitting and testing the SimpleLDM on the generated data
ldm = SimpleLDM()
ldm.fit(S[:-1].reshape(-1, 1))

# Predict the last time step
print(f"Actual value at last time step: {S[-1]}")
print(f"Predicted value at last time step: {ldm.predict(S[-2].reshape(-1, 1))}")

