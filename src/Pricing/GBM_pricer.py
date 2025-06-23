import numpy as np

class GBM:
    def __init__(self, S_0, mu, sigma, t, W_t):
        self.S_0 = S_0      # Initial stock price
        self.mu = mu        # Drift
        self.sigma = sigma  # Volatility
        self.t = t          # Time
        self.W_t = W_t      # Brownian motion value at time t

    def calculate_S_t(self):
        S_t = self.S_0 * np.exp((self.mu - 0.5 * self.sigma**2) * self.t + self.sigma * self.W_t)
        return S_t

# Example usage:
S_0 = 100
mu = 0.05
sigma = 0.2
t = 50
W_t = 0.00 # mean of Brownian motion is zero

gbm = GBM(S_0, mu, sigma, t, W_t)
S_t = gbm.calculate_S_t()
print(f"S(t) = {S_t}")
