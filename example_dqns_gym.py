"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from simulator.gui.multidrone_viewer_dqns_gym import MultiDroneViewerDQNS
from simulator.multidrone_gym_dqns import MultidroneGymDQNS
from simulator.utils.mobility_helper import grid_positions

dt = 0.1
num_drones = 25

size = 200.0

sim = MultidroneGymDQNS(num_drones, dt)
sim.environment.set_rectangular_boundary([0.0, 0.0], [size, size])

margin = 50.0
for _ in range(2):
    center = np.random.uniform(margin, +size, size=(2,))
    radius = np.random.uniform(5.0, 25.0)
    sim.environment.add_circular_obstacle(center, radius)

for _ in range(2):
    bottom_left = np.random.uniform(margin, +size, size=(2,))
    width_height = np.random.uniform(5.0, 25.0, size=(2,))
    top_right = bottom_left + width_height
    sim.environment.add_rectangular_obstacle(bottom_left, top_right)

margin = 5.0
p0 = grid_positions(num_drones, origin=[margin, margin], space=margin, altitude=0.0)
sim.initialize(positions=p0)

gui = MultiDroneViewerDQNS(sim)

while True:
    sim.update()
    gui.update(force_render=False, verbose=False)
    
    print(gui.viewer_status_str())
    print(sim.simulation_status_str())
    print(sim.training_status_str())
    if hasattr(sim, "rewards"):
        print("Rewards:", " ".join(f"{r:.2f}" for r in sim.rewards))