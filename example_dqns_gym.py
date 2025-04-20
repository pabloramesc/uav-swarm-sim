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

gui = MultiDroneViewerDQNS(sim, is_3d=False)

while True:
    metrics = sim.update()
    gui.update(force_render=False, verbose=False)

    coverage = sim.area_coverage_ratio()
    print(
        f"Sim time: {sim.time:.2f} s, "
        f"Area coverage: {coverage*100:.2f} %, "
        f"Train steps: {sim.dqns_agent.dqn_agent.train_steps}, "
        f"Memory size: {sim.dqns_agent.dqn_agent.memory.size}, ",
        end="",
    )
    if metrics:
        print(
            f"Loss: {metrics["loss"]:.4e}, "
            f"Accuracy: {metrics["accuracy"]*100:.4f} %"
        )
    else:
        print()
