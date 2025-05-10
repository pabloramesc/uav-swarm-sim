"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from multiagent_sim.multidrone_evsm_viewer import MultiDroneViewerEVSM
from multiagent_sim.multidrone_evsm_simulator import MultiDroneEVSMSimulator
from multiagent_sim.mobility.evsm_swarming import EVSMConfig
from multiagent_sim.utils.mobility_helper import grid_positions

dt = 0.1
num_drones = 50

evsm_config = EVSMConfig(
    separation_distance=50.0,
    obstacle_distance=10.0,
    max_acceleration=10.0,
    target_velocity=15.0,
    target_height=10.0,
)
sim = MultiDroneEVSMSimulator(num_drones, dt, config=evsm_config)
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
