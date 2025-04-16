"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.gui import MultiDroneViewerEVSM
from simulator.multidrone_simulator_evsm import MultiDroneSimulatorEVSM
from simulator.swarming import EVSMConfig
from simulator.utils.mobility_helper import grid_positions

dt = 0.1
num_drones = 100

config = EVSMConfig(separation_distance=1000.0, ln_rate=10.0)
sim = MultiDroneSimulatorEVSM(
    num_drones,
    dt,
    dem_path="./data/barcelona_dem.tif",
    config=config,
    visible_distance=1000.0,
)
sim.environment.set_rectangular_boundary([0e3, 0e3], [7e3, 7e3])
sim.environment.add_circular_obstacle([6e3, 3e3], 1e3)
sim.environment.add_rectangular_obstacle([0e3, 6e3], [5e3, 8e3])
p0 = grid_positions(num_drones, origin=[4e3, 4e3], space=5.0, altitude=0.0)
sim.initialize(positions=p0)

gui = MultiDroneViewerEVSM(sim, is_3d=True, aspect_ratio="auto", plot_regions_3d=False)

while True:
    sim.update()
    gui.update(force_render=False, verbose=True)
