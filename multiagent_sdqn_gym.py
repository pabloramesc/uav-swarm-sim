"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from multiagent_sim.multiagent_sdqn_viewer import MultiAgentSDQNViewer
from multiagent_sim.multiagent_sdqn_gym import MultiAgentSDQNGym
from multiagent_sim.mobility.sdqn_position_controller import SDQNPositionConfig

dt = 0.1
num_drones = 16

xy_min = (-200.0, -200.0)
xy_max = (+200.0, +200.0)

config = SDQNPositionConfig(num_cells=100, frame_radius=500.0, target_height=10.0)
sim = MultiAgentSDQNGym(
    num_drones, dt, config, model_path="data/models/sdqn-m01.keras", train=False
)
sim.environment.set_rectangular_boundary(xy_min, xy_max)

for _ in range(0):
    center = np.random.uniform(xy_min, xy_max, size=(2,))
    radius = np.random.uniform(5.0, 25.0)
    sim.environment.add_circular_obstacle(center, radius)

for _ in range(0):
    bottom_left = np.random.uniform(xy_min, xy_max, size=(2,))
    width_height = np.random.uniform(5.0, 50.0, size=(2,))
    top_right = bottom_left + width_height
    sim.environment.add_rectangular_obstacle(bottom_left, top_right)

sim.initialize()

gui = MultiAgentSDQNViewer(sim)


while True:
    sim.update()
    gui.update(force_render=False, verbose=False)

    print(gui.viewer_status_str())
    print(sim.simulation_status_str())
    print(sim.training_status_str())
    if hasattr(sim, "rewards"):
        print("Rewards:", " ".join(f"{r:.2f}" for r in sim.rewards))
    print()
