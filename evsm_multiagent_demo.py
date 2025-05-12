"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from multiagent_sim.multiagent_evsm_viewer import MultiAgentViewerEVSM
from multiagent_sim.multiagent_evsm_simulator import MultiAgentEVSMSimulator
from multiagent_sim.mobility.evsm_position_controller import EVSMPositionConfig

dt = 0.01
num_drones = 25
num_users = 10

evsm_config = EVSMPositionConfig(
    separation_distance=100.0,
    obstacle_distance=10.0,
    max_acceleration=10.0,
    target_speed=15.0,
    target_altitude=10.0,
    initial_natural_length=5.0,
    natural_length_rate=2.0,
)
sim = MultiAgentEVSMSimulator(
    num_drones, num_users, dt, evsm_config=evsm_config, neihgbor_provider="network"
)
sim.environment.set_rectangular_boundary([-200.0, -200.0], [+200.0, +200.0])
sim.environment.add_circular_obstacle([50.0, 50.0], 25.0)
sim.environment.add_rectangular_obstacle([-125.0, 0.0], [-100.0, +100.0])
sim.environment.add_rectangular_obstacle([50.0, -100.0], [100.0, -50.0])
sim.initialize(home=[-100.0, -100.0, 0.0])

gui = MultiAgentViewerEVSM(sim)

while True:
    sim.update()
    gui.update(force_render=False, verbose=True)

    if sim.sim_time % 10 == 0:
        sim.network_simulator.verify_node_positions()

    # cr = sim.area_coverage_ratio()
    # print(f"Area coverage ratio: {cr * 100:.2f} %")

    sim.sync_to_real_time()
