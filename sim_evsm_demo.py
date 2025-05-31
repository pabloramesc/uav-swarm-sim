"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from multiagent_sim.core.evsm_simulator import EVSMSimulator, EVSMConfig
from multiagent_sim.gui.evsm_viewer import EVSMViewer


dt = 0.01
num_drones = 25
num_users = 0
size = 1e3
grid_spacing = 10.0

evsm_config = EVSMConfig(
    separation_distance=350.0,
    obstacle_distance=20.0,
    max_acceleration=10.0,
    target_altitude=0.0,
    initial_natural_length=grid_spacing,
    natural_length_rate=5.0,
)
sim = EVSMSimulator(
    num_drones=num_drones,
    num_users=num_users,
    num_gcs=0,
    dt=dt,
    use_network=False,
    evsm_config=evsm_config,
)

sim.environment.set_rectangular_boundary([0, 0], [size, size])
sim.environment.add_circular_obstacle(center=[600, 600], radius=100)
sim.environment.add_rectangular_obstacle(bottom_left=[200, 600], top_right=[300, 800])
sim.environment.add_rectangular_obstacle(bottom_left=[600, 200], top_right=[800, 300])
sim.environment.add_rectangular_obstacle(bottom_left=[1e3, 1e3], top_right=[1e3, 1e3])

sim.initialize(home=[200, 200], spacing=grid_spacing)

gui = EVSMViewer(sim)

while True:
    sim.update()
    fps = gui.update(force=False)

    print(f"Real time: {sim.real_time:.2f} s, Sim time: {sim.sim_time:.2f} s, ", end="")
    if sim.network:
        print(f"NS-3 time: {sim.network.ns3_time:.2f} s, FPS: {gui.fps:.2f}")
    else:
        print(f"FPS: {gui.fps:.2f}")
