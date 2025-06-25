import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simulations.GBM_simulation import GBM as GBM_Simulation
from Pricing.GBM_pricer import GBM_formula as GBM_Pricer
import numpy as np

# Parameters
s_0 = 100
iterations = 1000  # T = iteration * dt = 1000 * 0.01 = 10
mu = 0.05
sigma = 0.2
num_paths = 100
t = 10
W_t = 0.0  # Mean of Brownian motion

# Run simulation
simulator = GBM_Simulation(s_0, iterations, num_paths, mu, sigma)
paths = simulator.monte_carlo()
mean_price = np.mean([path[-1] for path in paths])
print(f"Mean final stock price from simulation at t={t}: {mean_price}")

# Analytical solution
pricer = GBM_Pricer(s_0, mu, sigma, t, W_t)
S_t = pricer.calculate_S_t()
print(f"Analytical solution for S(t) at t={t}: {S_t}")
