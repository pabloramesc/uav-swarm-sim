"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from multiagent_sim.core.sdqn_trainer import SDQNTrainer, SDQNConfig
from multiagent_sim.gui.sdqn_viewer import SDQNViewer
from multiagent_sim.gui.sdqn_logpolar_viewer import SDQNLogPolarViewer

dt = 0.1
num_drones = 16
num_users = 20
size = 1e3
change_interval = 10_000
num_obstacles = 0

config = SDQNConfig(target_velocity=20.0, target_height=0.0)
sim = SDQNTrainer(
    num_drones,
    num_users,
    dt,
    sdqn_config=config,
    model_path="data/models/sdqn-m2e.keras",
    actions_mode="extended",
    train_mode=True,
    logpolar=False,
)

sim.environment.set_rectangular_boundary([-size, -size], [+size, +size])


def change_environment():
    sim.environment.clear_obstacles()

    for _ in range(num_obstacles):
        center = np.random.uniform(-size, +size, size=(2,))
        radius = np.random.uniform(0.02 * size, 0.2 * size)
        sim.environment.add_circular_obstacle(center, radius)

    for _ in range(num_obstacles):
        bottom_left = np.random.uniform(-size, +size, size=(2,))
        width_height = np.random.uniform(0.05 * size, 0.5 * size, size=(2,))
        top_right = bottom_left + width_height
        sim.environment.add_rectangular_obstacle(bottom_left, top_right)

    sim.initialize()


change_environment()

# gui = SDQNViewer(sim)
# gui = SDQNLogPolarViewer(sim)

while True:
    sim.update()

    if sim.sim_step % change_interval == 0:
        print("Changing environment...")
        change_environment()

    # fps = gui.update(force=False)

    # print(f"FPS: {gui.fps:.2f}")
    print(sim.simulation_status_str())
    print(sim.training_status_str())
    print("Rewards:", " ".join(f"{r:.2f}" for r in sim.rewards))
    print()
