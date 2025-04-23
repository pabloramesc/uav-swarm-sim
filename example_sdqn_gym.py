"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from simulator.gui.multidrone_viewer_sdqn_gym import MultiDroneViewerSDQN
from simulator.multidrone_gym_sdqn import MultidroneGymSDQN
from simulator.position_control.sdqn_position_control import SDQNConfig

dt = 0.1
num_drones = 16

xy_min = (-200.0, -200.0)
xy_max = (+200.0, +200.0)

config = SDQNConfig(num_cells=64, num_actions=9, visible_distance=100.0, target_height=0.0)
sim = MultidroneGymSDQN(num_drones, dt, config, model_path="sdqn-model-02.keras")
sim.environment.set_rectangular_boundary(xy_min, xy_max)

for _ in range(5):
    center = np.random.uniform(xy_min, xy_max, size=(2,))
    radius = np.random.uniform(5.0, 25.0)
    sim.environment.add_circular_obstacle(center, radius)

for _ in range(5):
    bottom_left = np.random.uniform(xy_min, xy_max, size=(2,))
    width_height = np.random.uniform(5.0, 50.0, size=(2,))
    top_right = bottom_left + width_height
    sim.environment.add_rectangular_obstacle(bottom_left, top_right)

sim.initialize()

gui = MultiDroneViewerSDQN(sim)

while True:
    sim.update()
    gui.update(force_render=False, verbose=False)
    
    print(gui.viewer_status_str())
    print(sim.simulation_status_str())
    print(sim.training_status_str())
    if hasattr(sim, "rewards"):
        print("Rewards:", " ".join(f"{r:.2f}" for r in sim.rewards))
    print()