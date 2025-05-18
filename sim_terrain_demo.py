"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from multiagent_sim.core.evsm_simulator import EVSMSimulator, EVSMConfig
from multiagent_sim.gui.simple_viewer import SimpleViewer


dt = 0.01
num_drones = 25
num_users = 10
size = 500.0

evsm_config = EVSMConfig(
    separation_distance=500.0,
    obstacle_distance=100.0,
    max_acceleration=10.0,
    target_speed=20.0,
    target_altitude=50.0,
    initial_natural_length=5.0,
    natural_length_rate=5.0,
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

vertices = [[1e3, 1e3], [6e3, 1e3], [9e3, 6e3], [9e3, 9e3], [1e3, 9e3]]
sim.environment.set_polygonal_boundary(vertices)

for _ in range(5):
    center = np.random.uniform(xy_min, xy_max, size=(2,))
    radius = np.random.uniform(50.0, 500.0)
    sim.environment.add_circular_obstacle(center, radius)

for _ in range(5):
    bottom_left = np.random.uniform(xy_min, xy_max, size=(2,))
    width_height = np.random.uniform(100.0, 1000.0, size=(2,))
    top_right = bottom_left + width_height
    sim.environment.add_rectangular_obstacle(bottom_left, top_right)

sim.initialize(home=[4.6e3, 1.5e3])

gui = SimpleViewer(sim, background_type="fused")

while True:
    sim.update()
    fps = gui.update(force=False)

    print(f"Real time: {sim.real_time:.2f} s, Sim time: {sim.sim_time:.2f} s, ", end="")
    if sim.network:
        print(f"NS-3 time: {sim.network.ns3_time:.2f} s, FPS: {gui.fps:.2f}")
    else:
        print(f"FPS: {gui.fps:.2f}")
