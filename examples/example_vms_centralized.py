"""
 Copyright (c) 2025 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from simulator.multidrone_simulator_evsm import MultiDroneSimulatorEVSM
from simulator.gui.multidrone_viewer_evsm import MultiDroneViewerEVSM

dt = 0.01
num_drones = 20
field_size = 1000.0

sim = MultiDroneSimulatorEVSM(num_drones, dt)
gui = MultiDroneViewerEVSM(num_drones, field_size)

sim.set_grid_positions()

while True:
    sim.update_central()
    gui.update(sim.drone_states, sim.links_matrix, sim.time, verbose=True)

