"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from multiagent_sim.core.evsm_simulator import EVSMSimulator, EVSMConfig
from multiagent_sim.gui import EVSMViewer


dt = 0.01
num_drones = 25
num_users = 10
size = 500.0

evsm_config = EVSMConfig(
    separation_distance=250.0,
    obstacle_distance=10.0,
    max_acceleration=10.0,
    target_speed=10.0,
    target_altitude=10.0,
    initial_natural_length=5.0,
    natural_length_rate=2.0,
)
sim = EVSMSimulator(
    num_drones,
    num_users,
    dt,
    use_network=True,
    evsm_config=evsm_config,
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

sim.initialize(home=[0.0, 0.0, 0.0])

gui = EVSMViewer(sim)

while True:
    sim.update()
    fps = gui.update(force=False)
    print(
        f"Real time: {sim.real_time:.2f} s, Sim time: {sim.sim_time:.2f} s, "
        f"NS-3 time: {sim.network.ns3_time:.2f} s, FPS: {gui.fps:.2f}"
    )
