"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import matplotlib.pyplot as plt
import numpy as np

from simulator.environment import CircularObstacle, Environment, RectangularBoundary
from simulator.swarming.sdqn import DQNS

# Define the environment
xlim = np.array([-200.0, +200.0])
ylim = np.array([-200.0, +200.0])
env = Environment(
    boundary=RectangularBoundary(
        (0.9 * xlim[0], 0.9 * ylim[0]), (0.9 * xlim[1], 0.9 * ylim[1])
    ),
    obstacles=[
        CircularObstacle(center=xy, radius=10.0)
        for xy in np.random.uniform((xlim[0], ylim[0]), (xlim[1], ylim[1]), (25, 2))
    ],
)

# Initialize DQNS
dqns = DQNS(env, num_cells=100, sense_radius=100.0)

# Define UAV position and neighbors
uav_position = np.array([0.0, 0.0])
neighbors = np.random.uniform((xlim[0], ylim[0]), (xlim[1], ylim[1]), (16, 2))

# Update DQNS
dqns.update(position=uav_position, neighbors=neighbors)

# Generate frame
frame = dqns.compute_state_frame()

# Plot the matrices and the real layout
fig, axes = plt.subplots(2, 2)

# Real Layout
axes[0, 0].set_title("Real Layout")
axes[0, 0].set_xlim(1.1 * xlim)
axes[0, 0].set_ylim(1.1 * ylim)
axes[0, 0].set_aspect("equal", adjustable="box")
axes[0, 0].set_xlabel("X")
axes[0, 0].set_ylabel("Y")

# Plot boundary
rect = plt.Rectangle(
    (env.boundary.bottom_left[0], env.boundary.bottom_left[1]),
    env.boundary.top_right[0] - env.boundary.bottom_left[0],
    env.boundary.top_right[1] - env.boundary.bottom_left[1],
    edgecolor="red",
    facecolor="none",
    linewidth=2,
)
axes[0, 0].add_artist(rect)

# Plot sense area
circle = plt.Circle(uav_position, dqns.sense_radius, color="red", alpha=0.1)
axes[0, 0].add_artist(circle)

# Plot obstacles
for obstacle in env.obstacles:
    circle = plt.Circle(obstacle.center, obstacle.radius, color="red", alpha=0.5)
    axes[0, 0].add_artist(circle)

# Plot UAV position
axes[0, 0].scatter(
    uav_position[0], uav_position[1], color="blue", label="UAV", zorder=5
)

# Plot neighbors
axes[0, 0].scatter(
    neighbors[:, 0], neighbors[:, 1], color="green", label="Neighbors", zorder=5
)

axes[0, 0].legend()
axes[0, 0].grid()

# Environment Matrix
environment_matrix = np.clip(np.sum(frame, axis=-1), 0, 255).astype("uint8")
axes[0, 1].imshow(environment_matrix, origin="lower")
axes[0, 1].set_title("Environment Matrix")
axes[0, 1].set_xlabel("X")
axes[0, 1].set_ylabel("Y")

# Collision Matrix
collision_matrix = frame[..., 0]
axes[1, 0].imshow(collision_matrix, origin="lower")
axes[1, 0].set_title("Collision Matrix")
axes[1, 0].set_xlabel("X")
axes[1, 0].set_ylabel("Y")

# # Distances Matrix
# distances_matrix = frame[..., 0]
# axes[1, 1].imshow(distances_matrix, origin="lower")
# axes[1, 1].set_title("Signal Matrix")
# axes[1, 1].set_xlabel("X")
# axes[1, 1].set_ylabel("Y")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
