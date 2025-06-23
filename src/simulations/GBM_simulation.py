import numpy as np
import matplotlib.pyplot as plt

class GBM:
    def __init__(self, s_0, iteration, num_paths, mu, sigma):
        # Initialize the GBM model parameters
        self.s_0 = s_0                  # Initial stock price
        self.iteration = iteration      # Number of time steps in each path
        self.num_paths = num_paths      # Number of simulated paths
        self.mu = mu                    # Expected return (drift)
        self.sigma = sigma              # Volatility of the stock

    def monte_carlo(self):
        dt = 0.01                       # Time increment
        all_paths = []                  # List to store all simulated paths
        for i in range(self.num_paths):
            prices = [self.s_0]         # Start each path with the initial stock price
            for j in range(self.iteration):
                s_t = prices[-1]        # Current stock price
                # Calculate the change in stock price using GBM formula
                ds = s_t * (self.mu * dt + self.sigma * np.sqrt(dt) * np.random.randn())
                prices.append(s_t + ds) # Append the new price to the current path
            all_paths.append(prices)    # Append the completed path to all_paths
        return all_paths

# Parameters
s_0 = 100
iterations = 5000 # T = iteration * dt = 5000 * 0.01 = 50
mu = 0.05
sigma = 0.2
num_paths = 100

# Create GBM instance with specified parameters
gbm = GBM(s_0, iterations, num_paths, mu, sigma)

# Simulate multiple GBM paths
paths = gbm.monte_carlo()

# Plot all simulated paths
plt.figure(figsize=(10, 6))
for path in paths:
    plt.plot(path)                     # Plot each simulated path
plt.title('Simulated GBM Paths')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()

# Extract the final stock price from each path
final_prices = [path[-1] for path in paths]

# Calculate the mean of the final stock prices
mean_final_price = np.mean(final_prices)

# Print the mean final price
print(f"The mean of the final stock prices from {len(paths)} simulated paths is: {mean_final_price}")
