"""
 Copyright (c) 2025 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from simulator.multidrone_evsm_simulator import MultiDroneEVSMSimulator
from simulator.gui.multidrone_viewer import MultiDroneViewer

dt = 0.01
num_drones = 20
field_size = 1000.0

sim = MultiDroneEVSMSimulator(num_drones, dt)
gui = MultiDroneViewer(num_drones, field_size)

sim.set_grid_positions()

while True:
    sim.update_central()
    gui.update(sim.drone_states, sim.links_matrix, sim.time, verbose=True)

