import numpy as np
import matplotlib.pyplot as plt

class GBM:
    def __init__(self, s_0, iteration, mu, sigma):
        self.s_0 = s_0
        self.iteration = iteration
        self.mu = mu
        self.sigma = sigma

    def monte_carlo(self):
        dt = 0.01
        prices = [self.s_0]

        for i in range(self.iteration):
            s_t = prices[-1]
            ds = s_t * (self.mu * dt + self.sigma * np.sqrt(dt) * np.random.randn())
            prices.append(s_t + ds)
        return prices


# Parameters
s_0 = 100
iterations = 1000
mu = 0.05
sigma = 0.2
num_paths = 10

# Create GBM instance
gbm = GBM(s_0, iterations, mu, sigma)

# Simulate multiple paths
paths = [gbm.monte_carlo() for i in range(num_paths)]

# Plot the results
plt.figure(figsize=(10, 6))
for path in paths:
    plt.plot(path)
plt.title('Simulated GBM Paths')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()