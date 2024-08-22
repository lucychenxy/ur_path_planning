import numpy as np
from scipy.interpolate import BSpline, splprep, splev
import matplotlib.pyplot as plt

# Generate a nonlinear trajectory in 6-dimensional space
n_points = 11
t = np.linspace(0, 2 * np.pi, n_points)

# Nonlinear functions for each dimension
x1 = np.sin(t)  # First dimension
x2 = np.cos(t)  # Second dimension
x3 = t+5  # Third dimension (linear)
x4 = np.sin(2 * t)  # Fourth dimension
x5 = np.log(t + 1)  # Fifth dimension
x6 = np.exp(0.1 * t)  # Sixth dimension

# Combine into a 6D trajectory
trajectory = np.vstack([x1, x2, x3, x4, x5, x6])

tck, u1 = splprep(trajectory, s=0, k=3)  #, u=u_manual
#tck, u2 = splprep(trajectory2, s=0, k=3)

# Generate new points on the B-spline
u_fine = np.linspace(0, 1, 100)  # Use more points for a smoother curve
new_points = splev(u_fine, tck)

# Convert the result to a numpy array for easier manipulation
new_points = np.array(new_points)

# Plotting the original and the interpolated trajectory
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

for i in range(6):
    ax = axes[i // 2, i % 2]
    ax.plot(t, trajectory[i], 'bo', label='Original points')
    ax.plot(np.linspace(0, 2 * np.pi, len(new_points[i])), new_points[i], 'r-', label='B-spline')
    ax.set_title(f'Dimension {i + 1}')
    ax.legend()

plt.tight_layout()
plt.show()