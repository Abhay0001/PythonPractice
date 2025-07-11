import numpy as np
import matplotlib.pyplot as plt

# Define matrix
A = np.array([[4, 2],
              [1, 3]])

# Eigen decomposition
eigvals, eigvecs = np.linalg.eig(A)

# Set of vectors to transform
vectors = [np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2*np.pi, 20)]

# Plot original and transformed vectors
plt.figure(figsize=(8, 8))
ax = plt.gca()

for v in vectors:
    v = v / np.linalg.norm(v)  # normalize
    Av = A @ v
    ax.quiver(*v, *(Av - v), angles='xy', scale_units='xy',
              scale=1, color='gray', alpha=0.5)

# Plot eigenvectors
for i in range(len(eigvals)):
    eigvec = eigvecs[:, i]
    eigvec = eigvec / np.linalg.norm(eigvec)
    scaled = eigvals[i] * eigvec
    ax.quiver(*eigvec, *(scaled - eigvec), angles='xy', scale_units='xy', scale=1,
              color='red' if i == 0 else 'blue', label=f'λ = {eigvals[i]:.2f}')

# Decorations
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_aspect('equal')
plt.grid(True)
plt.legend()
plt.title('Eigenvectors remain on their span, just scaled')
plt.show()
