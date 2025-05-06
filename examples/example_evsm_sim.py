"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from simulator.gui.multidrone_viewer_evsm import MultiDroneViewerEVSM
from simulator.multidrone_simulator_evsm import MultiDroneSimulatorEVSM
from simulator.position_control.evsm_position_control import EVSMConfig
from simulator.utils.mobility_helper import grid_positions

dt = 0.1
num_drones = 50

config = EVSMConfig(
    separation_distance=50.0,
    obstacle_distance=10.0,
    max_acceleration=10.0,
    target_velocity=15.0,
    target_height=10.0,
)
sim = MultiDroneSimulatorEVSM(num_drones, dt, config=config)
sim.environment.set_rectangular_boundary([-200.0, -100.0], [+200.0, +100.0])
sim.environment.add_circular_obstacle([25.0, 25.0], 25.0)
sim.environment.add_rectangular_obstacle([-125.0, -50.0], [-100.0, +50.0])
sim.environment.add_rectangular_obstacle([100.0, -50.0], [150.0, 0.0])
p0 = grid_positions(num_drones, origin=[-50.0, -50.0], space=5.0, altitude=0.0)
sim.initialize(positions=p0)

gui = MultiDroneViewerEVSM(sim, is_3d=False)

while True:
    sim.update()
    gui.update(force_render=False, verbose=True)

    cr = sim.area_coverage_ratio()
    print(f"Area coverage ratio: {cr * 100:.2f} %")
