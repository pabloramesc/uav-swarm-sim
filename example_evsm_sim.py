"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from simulator.gui import MultiDroneViewer
from simulator.multidrone_evsm_simulator import MultiDroneEVSMSimulator
from simulator.swarming import EVSMConfig

dt = 0.1
num_drones = 50

config = EVSMConfig(
    separation_distance=50.0,
    obstacle_distance=10.0,
    max_acceleration=10.0,
    target_velocity=15.0,
    target_altitude=10.0,
)
sim = MultiDroneEVSMSimulator(num_drones, dt, config=config)
sim.set_rectangular_boundary([-200.0, -100.0], [+200.0, +100.0])
sim.add_circular_obstacle([25.0, 25.0], 25.0)
sim.add_rectangular_obstacle([-125.0, -50.0], [-100.0, +50.0])
sim.add_rectangular_obstacle([100.0, -50.0], [150.0, 0.0])
sim.set_grid_positions(origin=[-50.0, -50.0], space=5.0)
sim.initialize()

gui = MultiDroneViewer(sim, is_3d=False)

while True:
    sim.update()
    gui.update(force=False, verbose=True)

    cr = sim.area_coverage_ratio()
    print(f"Area coverage ratio: {cr * 100:.2f} %")
