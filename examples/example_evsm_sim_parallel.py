"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.old.multidrone_simulator_parallel import MultiDroneSimulator
from simulator.gui.multidrone_viewer import MultiDroneViewer

if __name__ == "__main__":

    dt = 0.1
    num_drones = 2
    xlim = np.array([-200.0, +200.0])
    ylim = np.array([-100.0, +100.0])

    sim = MultiDroneSimulator(num_drones, dt)
    sim.set_rectangular_boundary((xlim[0], ylim[0]), (xlim[1], ylim[1]))
    sim.add_circular_obstacle((25.0, 25.0), 25.0)
    sim.add_rectangular_obstacle((-125.0, -50.0), (-100.0, +50.0))
    sim.add_rectangular_obstacle((100.0, -50.0), (150.0, 0.0))
    sim.initialize_grid_positions(origin=[-50.0, -50.0], space=5.0)

    gui = MultiDroneViewer(sim, xlim * 1.1, ylim * 1.1)

    sim.launch_simulation()

    while True:
        sim.update()
        gui.update(force_render=False, verbose=True)

        if gui.non_render_steps == 0:
            speed = np.linalg.norm(sim.drone_velocities[0, 0:2])
            print(f"Drone 0 speed: {speed:.2f} m/s")
