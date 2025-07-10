import numpy as np
import matplotlib.pyplot as plt

'''
Extended Hull-White is an extension of 2 factor Hull-White process to 3 factor.
Each component follows its own stochastic differential equation (SDE):
dx(t) = flt_speed1 * (flt_long_mean1 - x(t) - flt_coupling2 * y(t) - flt_coupling3 * z(t)) * dt + flt_vol1 * sqrt(dt) * z1
dy(t) = -flt_speed2 * y(t) * dt + flt_vol2 * sqrt(dt) * z2
dz(t) = -flt_speed3 * z(t) * dt + flt_vol3 * sqrt(dt) * z3

Where:
- flt_speed1, flt_speed2, flt_speed3: Mean reversion speeds (how quickly the process reverts to its long-term mean)
- flt_long_mean1, 0, 0: Long-term means of the stochastic processes dx, dy, dz
- sigma1, sigma2, sigma3: Volatilities of the processes
- z1, z2, z3: Standard normal random variables (with specified correlations)
The final increment in interest rate is dx, since y and z are inside x. 
The coupling coefficient flt_coupling2 and flt_coupling3 are included to add generality to the model.
This can control the effect of y(t) and z(t) on x(t).
'''


class HullWhite3FactorCoupled:
    def __init__(self, flt_speed1, flt_speed2, flt_speed3, flt_long_mean1, flt_coupling2, flt_coupling3, flt_vol1, flt_vol2, flt_vol3):
        self.flt_speed1 = flt_speed1
        self.flt_speed2 = flt_speed2
        self.flt_speed3 = flt_speed3
        self.flt_long_mean1 = flt_long_mean1
        self.flt_coupling2 = flt_coupling2
        self.flt_coupling3 = flt_coupling3
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
        np.random.seed(42)

        for i in range(num_of_paths):
            # Initial rate at t=0
            x = [x0]
            y = [y0]
            z = [z0]
            r = [x0]

            for j in range(iterations):
                z_corr = self.cholesky_correlated_normals(rho12, rho13, rho23)

                dy = -self.flt_speed2 * (y[-1]) * dt + self.flt_vol2 * np.sqrt(dt) * z_corr[1]
                dz = -self.flt_speed3 * (z[-1]) * dt + self.flt_vol3 * np.sqrt(dt) * z_corr[2]

                y_new = y[-1] + dy
                z_new = z[-1] + dz

                dx = self.flt_speed1 * (self.flt_long_mean1 - x[-1] + self.flt_coupling2 * y_new + self.flt_coupling3 * z_new) * dt + self.flt_vol1 * np.sqrt(dt) * z_corr[0]
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

# Example
model = HullWhite3FactorCoupled(
    flt_speed1=0.3, flt_speed2=0.7, flt_speed3=1.1,
    flt_long_mean1=0.002,
    flt_coupling2=0.8, flt_coupling3=0.6,
    flt_vol1=0.15, flt_vol2=0.25, flt_vol3=0.35
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
