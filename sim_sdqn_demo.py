"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from multiagent_sim.core.sdqn_simulator import SDQNSimulator, SDQNConfig
from multiagent_sim.gui.sdqn_viewer import SDQNViewer

dt = 0.1
num_drones = 16
num_users = 20
size = 500.0
num_obstacles = 0

config = SDQNConfig(target_velocity=20.0, target_height=10.0)
sim = SDQNSimulator(
    num_drones,
    num_users,
    dt,
    sdqn_config=config,
    model_path="data/models/sdqn-m1c.keras",
    actions_mode="extended",
    train_mode=False,
)

sim.environment.set_rectangular_boundary([-size, -size], [+size, +size])

for _ in range(num_obstacles):
    center = np.random.uniform(-size, +size, size=(2,))
    radius = np.random.uniform(0.01 * size, 0.1 * size)
    sim.environment.add_circular_obstacle(center, radius)

for _ in range(num_obstacles):
    bottom_left = np.random.uniform(-size, +size, size=(2,))
    width_height = np.random.uniform(0.05 * size, 0.5 * size, size=(2,))
    top_right = bottom_left + width_height
    sim.environment.add_rectangular_obstacle(bottom_left, top_right)

sim.initialize()

gui = SDQNViewer(sim)

while True:
    sim.update()
    fps = gui.update(force=False)

    print(f"FPS: {gui.fps:.2f}")
    print(sim.simulation_status_str())
    print(sim.training_status_str())
    print("Rewards:", " ".join(f"{r:.2f}" for r in sim.rewards))
    print()
