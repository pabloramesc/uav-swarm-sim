"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from multiagent_sim.core.evsm_simulator import EVSMSimulator, EVSMConfig
from multiagent_sim.gui.simple_viewer import SimpleViewer


dt = 0.01
grid_spacing = 100.0
num_drones = 25
num_users = 50

evsm_config = EVSMConfig(
    separation_distance=1e3,
    obstacle_distance=100.0,
    target_speed=20.0,
    target_altitude=50.0,
    initial_natural_length=grid_spacing,
    natural_length_rate=10.0,
    max_acceleration=10.0,
    max_position_error=100.0,
)
sim = EVSMSimulator(
    num_drones,
    num_users,
    dt,
    use_network=False,
    evsm_config=evsm_config,
    dem_path="data/elevation/barcelona_dem.tif",
)

xy_min = [1e3, 1e3]
xy_max = [9e3, 9e3]
# sim.environment.set_rectangular_boundary(xy_min, xy_max)

vertices = [[2e3, 2e3], [6e3, 2e3], [8e3, 4e3], [8e3, 8e3], [2e3, 8e3]]
sim.environment.set_polygonal_boundary(vertices)
sim.environment.add_circular_obstacle([6e3, 4e3], 0.5e3)
sim.environment.add_rectangular_obstacle([4.0e3, 5.5e3], [6.0e3, 6.5e3])

sim.initialize(home=[4e3, 4e3], spacing=grid_spacing, altitude=100.0)

gui = SimpleViewer(sim, background_type="fused")

while True:
    sim.update()
    fps = gui.update(force=False)

    print(f"Real time: {sim.real_time:.2f} s, Sim time: {sim.sim_time:.2f} s, ", end="")
    if sim.network:
        print(f"NS-3 time: {sim.network.ns3_time:.2f} s, FPS: {gui.fps:.2f}")
    else:
        print(f"FPS: {gui.fps:.2f}")
