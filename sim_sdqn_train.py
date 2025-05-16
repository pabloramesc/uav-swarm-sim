"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")

from multiagent_sim.mobility.sdqn_position_controller import SDQNConfig
from multiagent_sim.old.multiagent_sdqn_gym import MultiAgentSDQNGym
from multiagent_sim.gui.sdqn_viewer import MultiAgentSDQNViewer


dt = 0.1
num_drones = 16
num_users = 10
size = 500.0

config = SDQNConfig(displacement=10.0, target_height=10.0)
sim = MultiAgentSDQNGym(
    num_drones,
    num_users,
    dt,
    sdqn_config=config,
    model_path="data/models/sdqn-m01.keras",
)

sim.environment.set_rectangular_boundary([-size, -size], [+size, +size])

for _ in range(5):
    center = np.random.uniform(-size, +size, size=(2,))
    radius = np.random.uniform(5.0, 50.0)
    sim.environment.add_circular_obstacle(center, radius)

for _ in range(5):
    bottom_left = np.random.uniform(-size, +size, size=(2,))
    width_height = np.random.uniform(10.0, 100.0, size=(2,))
    top_right = bottom_left + width_height
    sim.environment.add_rectangular_obstacle(bottom_left, top_right)

sim.initialize()

# gui = MultiAgentSDQNViewer(sim)

while True:
    sim.update()
    # gui.update(force_render=False, verbose=False)

    # print(gui.viewer_status_str())
    print(sim.simulation_status_str())
    print(sim.training_status_str())
    if hasattr(sim, "rewards"):
        print("Rewards:", " ".join(f"{r:.2f}" for r in sim.rewards))
    print()
