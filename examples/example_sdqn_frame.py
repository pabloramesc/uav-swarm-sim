"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import matplotlib.pyplot as plt
import numpy as np

from simulator.environment import CircularObstacle, Environment, RectangularBoundary
from simulator.sdqn.frame_generator import FrameGenerator

# Define the environment
xlim = np.array([-200.0, +200.0])
ylim = np.array([-200.0, +200.0])
env = Environment(
    boundary=RectangularBoundary(
        (0.9 * xlim[0], 0.9 * ylim[0]), (0.9 * xlim[1], 0.9 * ylim[1])
    ),
    obstacles=[
        CircularObstacle(center=xy, radius=10.0)
        for xy in np.random.uniform((xlim[0], ylim[0]), (xlim[1], ylim[1]), (20, 2))
    ],
)

# Initialize SDQN
sdqn = FrameGenerator(env, num_cells=100, sense_radius=100.0)

# Define UAV position and neighbors
uav_position = np.array([0.0, 0.0])
neighbors = np.random.uniform((xlim[0], ylim[0]), (xlim[1], ylim[1]), (16, 2))

# Update SDQN
sdqn.update(position=uav_position, neighbors=neighbors)

# Generate frame
frame = sdqn.compute_state_frame()

# Plot the matrices and the real layout
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# Real Layout
ax1.set_title("Real Layout")
ax1.set_xlim(1.1 * xlim)
ax1.set_ylim(1.1 * ylim)
ax1.set_aspect("equal", adjustable="box")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

# Plot boundary
rect = plt.Rectangle(
    (env.boundary.bottom_left[0], env.boundary.bottom_left[1]),
    env.boundary.top_right[0] - env.boundary.bottom_left[0],
    env.boundary.top_right[1] - env.boundary.bottom_left[1],
    edgecolor="red",
    facecolor="none",
    linewidth=2,
)
ax1.add_artist(rect)

# Plot sense area
circle = plt.Circle(uav_position, sdqn.sense_radius, color="red", alpha=0.1)
ax1.add_artist(circle)

# Plot obstacles
for obstacle in env.obstacles:
    circle = plt.Circle(obstacle.center, obstacle.radius, color="red", alpha=0.5)
    ax1.add_artist(circle)

# Plot UAV position
ax1.scatter(uav_position[0], uav_position[1], color="blue", label="UAV", zorder=5)

# Plot neighbors
ax1.scatter(
    neighbors[:, 0], neighbors[:, 1], color="green", label="Neighbors", zorder=5
)

ax1.legend()
ax1.grid()

# Collision heatmap
im2 = ax2.imshow(frame[..., 0] / 255.0, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
ax2.set_title("Collision heatmap")
ax2.set_xlabel("X (px)")
ax2.set_ylabel("Y (px)")
fig.colorbar(im2, ax=ax2)

# Coverage heatmap
im3 = ax3.imshow(
    (frame[..., 1] / 127.5) - 1.0, cmap="bwr", origin="lower", vmin=-1.0, vmax=+1.0
)
ax3.set_title("Coverage heatmap")
ax3.set_xlabel("X (px)")
ax3.set_ylabel("Y (px)")
fig.colorbar(im3, ax=ax3)

# Motion map
ax4.imshow(frame[..., 2], origin="lower")
ax4.set_title("Motion map")
ax4.set_xlabel("X (px)")
ax4.set_ylabel("Y (px)")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
