"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.multidrone_simulator import MultiDroneSimulator
from simulator.gui.multidrone_viewer import MultiDroneViewer

dt = 0.1
num_drones = 100
xlim = np.array([2e3, 14e3])
ylim = np.array([1e3, 8e3])

sim = MultiDroneSimulator(num_drones, dt, dem_path="./data/barcelona_dem.tif")
sim.set_rectangular_boundary((xlim[0], ylim[0]), (xlim[1], ylim[1]))
sim.add_circular_obstacle((10e3, 4e3), 2e3)
sim.add_rectangular_obstacle((2e3, 6e3), (6e3, 10e3))
sim.set_grid_positions(origin=[4e3, 4e3], space=5.0)
sim.initialize()

gui = MultiDroneViewer(sim)

while True:
    sim.update()        
    gui.update(force=False, verbose=True)
