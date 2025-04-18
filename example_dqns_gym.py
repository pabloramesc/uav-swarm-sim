"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from simulator.gui.multidrone_viewer_dqns import MultiDroneViewerDQNS
from simulator.multidrone_gym_dqns import MultidroneGymDQNS
from simulator.utils.mobility_helper import grid_positions

dt = 0.1
num_drones = 25

size = 200.0

sim = MultidroneGymDQNS(num_drones, dt)
sim.environment.set_rectangular_boundary([0.0, 0.0], [size, size])

margin = 50.0
for _ in range(5):
    center = np.random.uniform(margin, +size, size=(2,))
    radius = np.random.uniform(5.0, 25.0)
    sim.environment.add_circular_obstacle(center, radius)
    
for _ in range(5):
    bottom_left = np.random.uniform(margin, +size, size=(2,))
    width_height = np.random.uniform(5.0, 25.0, size=(2,))
    top_right = bottom_left + width_height
    sim.environment.add_rectangular_obstacle(bottom_left, top_right)

margin = 5.0
p0 = grid_positions(num_drones, origin=[margin, margin], space=margin, altitude=0.0)
sim.initialize(positions=p0)

gui = MultiDroneViewerDQNS(sim, is_3d=False)

while True:
    metrics = sim.update()
    gui.update(force_render=False, verbose=True)
    
    if metrics:
        print(f"Train steps: {metrics["train_steps"]}")

    cr = sim.area_coverage_ratio()
    print(f"Area coverage ratio: {cr * 100:.2f} %")
