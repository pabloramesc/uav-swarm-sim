"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from multiagent_sim.gui.evsm_viewer import EVSMViewer
from multiagent_sim.core.evsm_simulator import EVSMSimulator
from multiagent_sim.mobility.evsm_position_controller import EVSMConfig

dt = 0.1
num_drones = 25
num_users = 50

config = EVSMConfig(separation_distance=1000.0, natural_length_rate=10.0)
sim = EVSMSimulator(
    num_drones,
    num_users,
    dt,
    dem_path="./data/elevation/barcelona_dem.tif",
    use_network=False,
    evsm_config=config,
)
sim.environment.set_rectangular_boundary([0e3, 0e3], [7e3, 7e3])
sim.environment.add_circular_obstacle([6e3, 3e3], 1e3)
sim.environment.add_rectangular_obstacle([0e3, 6e3], [5e3, 8e3])
sim.initialize(home=[4e3, 4e3])

gui = EVSMViewer(sim, background_type="fused")

while True:
    sim.update()
    gui.update(force_render=False, verbose=True)
