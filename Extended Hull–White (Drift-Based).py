import numpy as np
import matplotlib.pyplot as plt

class HullWhite3FactorCoupled:
    def __init__(self, a1, a2, a3, mu1, beta2, beta3, sigma1, sigma2, sigma3):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.mu1 = mu1
        self.beta2 = beta2
        self.beta3 = beta3
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3

    def cholesky_correlated_normals(self, rho12, rho13, rho23):
        corr_matrix = np.array([
            [1.0, rho12, rho13],
            [rho12, 1.0, rho23],
            [rho13, rho23, 1.0]
        ])
        L = np.linalg.cholesky(corr_matrix)
        z_uncorr = np.random.normal(0, 1, 3)
        z_corr = np.dot(L, z_uncorr)
        return z_corr

    def MC_model(self, num_of_paths, iterations, x0, y0, z0, rho12, rho13, rho23):
        dt = 0.01
        all_paths = []
        np.random.seed(42)

        for i in range(num_of_paths):
            x = [x0]
            y = [y0]
            z = [z0]
            r = [x0]

            for j in range(iterations):
                z_corr = self.cholesky_correlated_normals(rho12, rho13, rho23)

                dy = -self.a2 * (y[-1]) * dt + self.sigma2 * np.sqrt(dt) * z_corr[1]
                dz = -self.a3 * (z[-1]) * dt + self.sigma3 * np.sqrt(dt) * z_corr[2]

                y_new = y[-1] + dy
                z_new = z[-1] + dz

                dx = self.a1 * (self.mu1 - x[-1] + self.beta2 * y_new + self.beta3 * z_new) * dt + self.sigma1 * np.sqrt(dt) * z_corr[0]
                x_new = x[-1] + dx

                x.append(x_new)
                y.append(y_new)
                z.append(z_new)
                r.append(x_new)

            all_paths.append(r)

        return all_paths

    def mean_path(self, all_paths):
        final_rates = [path[-1] for path in all_paths]
        return np.mean(final_rates)

# Example usage
model = HullWhite3FactorCoupled(
    a1=0.3, a2=0.7, a3=1.1,
    mu1=0.002,
    beta2=0.8, beta3=0.6,
    sigma1=0.15, sigma2=0.25, sigma3=0.35
)

simulated_paths = model.MC_model(
    num_of_paths=10,
    iterations=100,
    x0=0.01, y0=0.01, z0=0.01,
    rho12=0.2, rho13=0.1, rho23=0.3
)

mean_final_rate = model.mean_path(simulated_paths)
print(f"Mean of final interest rates across all paths: {mean_final_rate:.6f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.xlabel('Time Step')
plt.ylabel('Interest Rate')
plt.title('Simulated Interest Rate Paths (Generalized 3-Factor Hull-White Model)')

for path in simulated_paths:
    plt.plot(path, linewidth=0.8)

plt.grid(True)
plt.tight_layout()
plt.show()
