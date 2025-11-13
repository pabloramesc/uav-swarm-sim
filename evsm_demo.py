"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from sim.simulators.evsm_simulator import EVSMSimulator, EVSMConfig
from sim.gui.evsm_viewer import EVSMViewer
from sim.utils.data_logger import DataLogger
from sim.environment import Environment

dt = 0.01
num_drones = 25
num_users = 10
size = 1e3
grid_spacing = 10.0

config = EVSMConfig(
    separation_distance=350.0,
    obstacle_distance=20.0,
    max_acceleration=10.0,
    target_altitude=0.0,
    initial_natural_length=grid_spacing,
    natural_length_rate=5.0,
)

env = Environment()
env.set_rectangular_boundary([0, 0], [size, size])
env.add_circular_obstacle(center=[600, 600], radius=100)
env.add_rectangular_obstacle(bottom_left=[200, 600], top_right=[300, 800])
env.add_rectangular_obstacle(bottom_left=[600, 200], top_right=[800, 300])
env.add_rectangular_obstacle(bottom_left=[1e3, 1e3], top_right=[1e3, 1e3])

sim = EVSMSimulator(
    environment=env,
    num_drones=num_drones,
    num_users=num_users,
    num_gcs=1,
    dt=dt,
    use_network=False,
    evsm_config=config,
)

sim.reset(
    home=np.array([200, 200]), spacing=grid_spacing, altitude=config.target_altitude
)

gui = EVSMViewer(sim, show_legend=True)

log = DataLogger(
    log_file="log_evsm_network.npz",
    log_folder="logs",
    columns=[
        "time",
        "area_cov",
        "users_cov",
        "direct_conn",
        "global_conn",
        "send_packets",
        "recv_packets",
    ],
)


while sim.sim.clock.sim_time <= 120.0:
    sim.step()
    fps = gui.render()

    send_packets = (
        sum(user.swarm_link.send_counter for user in sim.sim.users)
        if sim.sim.network
        else 0
    )
    recv_packets = (
        sum(user.swarm_link.recv_counter for user in sim.sim.users)
        if sim.sim.network
        else 0
    )

    log.append(
        [
            sim.sim.clock.sim_time,
            sim.sim.metrics.area_coverage,
            sim.sim.metrics.users_coverage,
            sim.sim.metrics.direct_connections,
            sim.sim.metrics.global_connections,
            send_packets / num_users,
            recv_packets / num_users,
        ]
    )

    print(
        f"Real time: {sim.sim.clock.real_time:.2f} s, Sim time: {sim.sim.sim_time:.2f} s, ",
        end="",
    )
    if sim.sim.network:
        print(
            f"NS-3 time: {sim.sim.network.ns3_time:.2f} s, FPS: {gui.fps:.2f}", end="\r"
        )
    else:
        print(f"FPS: {gui.fps:.2f}", end="\r")
