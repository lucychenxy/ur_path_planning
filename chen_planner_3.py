import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline

init_degrees = np.array([163, -65, 34, -69, -91, 134])
init_rad = np.deg2rad(init_degrees)

final_degrees = np.array([0.2, -66, 79, -121, -91, 134])
final_rad = np.deg2rad(final_degrees)

tolerance_init = np.deg2rad(np.array([2, 2, 2, 2, 2, 2]))
tolerance_final = np.deg2rad(np.array([2, 2, 2, 2, 2, 2]))

num_control_points = 8
num_trajectories = 200

all_trajectories = []

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")


for i in range(num_trajectories):
    init_rad = init_rad + np.random.uniform(-tolerance_init, tolerance_init)
    final_rad = final_rad + np.random.uniform(-tolerance_final, tolerance_final)

    middle_control_points = np.linspace(init_rad, final_rad, num_control_points + 2)[
        1:-1
    ]
    middle_control_points += np.random.normal(
        scale=0.05, size=middle_control_points.shape
    )

    control_points = np.vstack([init_rad, middle_control_points, final_rad])

    t = np.linspace(0, 1, 11)
    spline = make_interp_spline(
        np.linspace(0, 1, num_control_points + 2), control_points, k=3
    )
    trajectory = spline(t)

    if not np.allclose(trajectory[0], init_rad) or not np.allclose(
        trajectory[-1], final_rad
    ):
        print(
            f"Warning: Trajectory {i+1} does not start at the initial point or end at the final point."
        )
        continue

    all_trajectories.append(np.rad2deg(trajectory))

    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.6)

ax.set_title("3D Continuous Trajectories from Initial to Final Positions")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")

plt.show()

for i, trajectory in enumerate(all_trajectories):
    df = pd.DataFrame(
        trajectory, columns=[f"Dim_{j+1}" for j in range(trajectory.shape[1])]
    )
    df.to_csv(f"trajectory_{i+1}.csv", index=False)

print("All trajectories have been saved to CSV files.")

np.save("all_trajectories.npy", np.array(all_trajectories))

print("All trajectories have been saved to 'all_trajectories.npy'.")
