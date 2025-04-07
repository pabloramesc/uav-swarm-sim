"""
 Copyright (c) 2025 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from simulator.multidrone_simulator import MultiDroneSimulator
from simulator.multidrone_viewer import MultiDroneViewer

dt = 0.01
num_drones = 20
field_size = 1000.0

sim = MultiDroneSimulator(num_drones, dt)
gui = MultiDroneViewer(num_drones, field_size)

sim.initialize_grid_positions()

while True:
    sim.update_central()
    gui.update(sim.drone_states, sim.links_matrix, sim.time, verbose=True)

