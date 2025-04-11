"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.multidrone_simulator import MultiDroneSimulator
from simulator.gui.multidrone_viewer import MultiDroneViewer

dt = 0.1
num_drones = 60
xlim = np.array([-200.0, +200.0])
ylim = np.array([-100.0, +100.0])

sim = MultiDroneSimulator(num_drones, dt)
sim.set_rectangular_boundary((xlim[0], ylim[0]), (xlim[1], ylim[1]))
sim.add_circular_obstacle((25.0, 25.0), 25.0)
sim.add_rectangular_obstacle((-125.0, -50.0), (-100.0, +50.0))
sim.add_rectangular_obstacle((100.0, -50.0), (150.0, 0.0))
sim.set_grid_positions(origin=[-50.0, -50.0], space=5.0)
sim.initialize()

gui = MultiDroneViewer(sim, xlim * 1.1, ylim * 1.1)

while True:
    sim.update()        
    gui.update(force_render=False, verbose=True)

    if gui.non_render_steps == 0:
        speed = np.linalg.norm(sim.drones[0].velocity)
        print(f"Drone 0 speed: {speed:.2f} m/s")
