import numpy as np
import random
import matplotlib.pyplot as plt

class HullWhite3Factor:
    def __init__(self, a1, a2, a3, mu_1, mu_2, mu_3, sigma1, sigma2, sigma3):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.mu_3 = mu_3
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3

    def MC_model(self, num_of_paths, iterations, x0, y0, z0, rho12, rho13, rho23):
        dt = 0.01
        all_paths = []

        # Correlation matrix and Cholesky decomposition
        corr_matrix = np.array([
            [1.0, rho12, rho13],
            [rho12, 1.0, rho23],
            [rho13, rho23, 1.0]
        ])
        L = np.linalg.cholesky(corr_matrix)

        np.random.seed(10)  # Set seed once for reproducibility

        for i in range(num_of_paths):
            x = [x0]
            y = [y0]
            z = [z0]
            r = [x0 + y0 + z0]

            for j in range(iterations):
                z_uncorr = np.random.normal(0, 1, 3)
                z_corr = np.dot(L, z_uncorr)

                dx = self.a1 * ( self.mu_1 - x[-1]) * dt + self.sigma1 * np.sqrt(dt) * z_corr[0]
                dy =  self.a2 * ( self.mu_2 - y[-1]) * dt + self.sigma2 * np.sqrt(dt) * z_corr[1]
                dz =  self.a3 * ( self.mu_3 - z[-1]) * dt + self.sigma3 * np.sqrt(dt) * z_corr[2]

                x.append(x[-1] + dx)
                y.append(y[-1] + dy)
                z.append(z[-1] + dz)
                r.append(x[-1] + y[-1] + z[-1])

            all_paths.append(r)

        return all_paths

    def mean_path(self, all_paths):
        final_rates = [path[-1] for path in all_paths]
        return np.mean(final_rates)


# Instantiate the model
model = HullWhite3Factor(a1=0.3, a2=0.7, a3=1.1, mu_1=0.002, mu_2=0.004, mu_3=0.006, sigma1=0.015, sigma2=0.025,
                         sigma3=0.035)

# Simulate paths
simulated_paths = model.MC_model(
    num_of_paths=10,
    iterations=100,
    x0=0.01,
    y0=0.01,
    z0=0.01,

    rho12=0.2,
    rho13=0.1,
    rho23=0.3
)

# Calculate mean of final interest rates
mean_final_rate = model.mean_path(simulated_paths)
print(f"Mean of final interest rates across all paths: {mean_final_rate:.6f}")

# Plot the simulated paths
plt.figure(figsize=(10, 6))
plt.xlabel('Time Step')
plt.ylabel('Interest Rate')
plt.title('Simulated Interest Rate Paths (Hull-White 3-Factor Model)')

for path in simulated_paths:
    plt.plot(path, linewidth=0.8)

plt.grid(True)
plt.tight_layout()
plt.show()
