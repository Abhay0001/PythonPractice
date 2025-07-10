import numpy as np
import matplotlib.pyplot as plt

'''
Additive Ornstein-Uhlenbeck process is one way to construct a 3-factor Hull-White model.
This model uses 3 Ornstein-Uhlenbeck processes (which are essentially one-factor Hull-White models).

dx(t) = flt_speed1(flt_long_mean1 - x(t))dt + flt_vol1*sqrt(dt)*z1  (z1 ~ N(0,1))
dy(t) = flt_speed2(flt_long_mean2 - y(t))dt + flt_vol2*sqrt(dt)*z2  (z2 ~ N(0,1))
dz(t) = flt_speed3(flt_long_mean3 - z(t))dt + flt_vol3*sqrt(dt)*z3  (z3 ~ N(0,1))

The final increment in interest rate dr(t) = dx(t) + dy(t) + dz(t), where z1, z2, z3 are correlated.

a_i = relaxation/mean reversion speed (how fast the process reverts to its mean)
mu_i = long-term mean of the process
sigma_i = standard deviation of the process
z_i = standard normal random variable ~ N(0,1)
i = 1, 2, 3 for x, y, z respectively
'''

class HullWhite3Factor:
    def __init__(self, flt_speed1, flt_speed2, flt_speed3, flt_long_mean1, flt_long_mean2, flt_long_mean3, flt_vol1, flt_vol2, flt_vol3):
        self.flt_speed1 = flt_speed1
        self.flt_speed2 = flt_speed2
        self.flt_speed3 = flt_speed3
        self.flt_long_mean1 = flt_long_mean1
        self.flt_long_mean2 = flt_long_mean2
        self.flt_long_mean3 = flt_long_mean3
        self.flt_vol1 = flt_vol1
        self.flt_vol2 = flt_vol2
        self.flt_vol3 = flt_vol3

    def cholesky_correlated_normals(self, rho12, rho13, rho23):
        '''
        This function generates three correlated standard normal random variables
        using the Cholesky decomposition method.

        Arguments:
        - rho12 (float): Correlation coefficient between variable 1 and 2.
        - rho13 (float): Correlation coefficient between variable 1 and 3.
        - rho23 (float): Correlation coefficient between variable 2 and 3.

        Operation:
        - Constructs a 3x3 correlation matrix using the provided correlation coefficients.
        - Applies Cholesky decomposition to obtain a lower triangular matrix `L`.
        - Generates three independent standard normal random variables.
        - Multiplies the Cholesky matrix `L` with the uncorrelated variables to produce
          correlated standard normal variables.
        - Returns a NumPy array of three correlated normal variables.
        '''
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

        np.random.seed(10)  # Set seed once for reproducibility
        for i in range(num_of_paths):
            # # Initial rates at t=0
            x = [x0]
            y = [y0]
            z = [z0]
            r = [x0 + y0 + z0]

            for j in range(iterations):
                z_corr = self.cholesky_correlated_normals(rho12, rho13, rho23)

                dx = self.flt_speed1 * (self.flt_long_mean1 - x[-1]) * dt + self.flt_vol1 * np.sqrt(dt) * z_corr[0]
                dy = self.flt_speed2 * (self.flt_long_mean2 - y[-1]) * dt + self.flt_vol2 * np.sqrt(dt) * z_corr[1]
                dz = self.flt_speed3 * (self.flt_long_mean3 - z[-1]) * dt + self.flt_vol3 * np.sqrt(dt) * z_corr[2]

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
model = HullWhite3Factor(
    flt_speed1=0.3, flt_speed2=0.7, flt_speed3=1.1,
    flt_long_mean1=0.002, flt_long_mean2=0.003, flt_long_mean3=0.004,
    flt_vol1=0.15, flt_vol2=0.25, flt_vol3=0.35
)

# Simulate paths
simulated_paths = model.MC_model(
    num_of_paths=10,
    iterations=100,
    x0=0.01, y0=0.01, z0=0.01,
    rho12=0.2, rho13=0.1, rho23=0.3
)

# Calculate mean of final interest rates
mean_final_rate = model.mean_path(simulated_paths)
print(f"Mean of final interest rates across all paths: {mean_final_rate:.6f}")

# Plot the simulated paths
plt.figure(figsize=(10, 6))
plt.xlabel('Time Step')
plt.ylabel('Interest Rate')
plt.title('Simulated Interest Rate Paths (3-Factor Hull-White Model (Additive OU))')

for path in simulated_paths:
    plt.plot(path, linewidth=0.8)

plt.grid(True)
plt.tight_layout()
plt.show()
