"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from multiagent_sim.core.sdqn_trainer import SDQNTrainer, SDQNConfig
from multiagent_sim.gui.sdqn_viewer import SDQNViewer
from multiagent_sim.gui.sdqn_logpolar_viewer import SDQNLogPolarViewer
from multiagent_sim.utils.csv_logger import CSVLogger
from multiagent_sim.sdqn.frames import (
    FrameGeneratorFactory,
    SquareGeometryFactory,
    SignalLayerFactory,
    get_neighbor_positions,
    get_user_positions,
)


dt = 0.1
num_drones = 16
num_users = 0
size = 1e3
num_obstacles = 0
num_episodes = 1000
max_steps = int(5 * 60 / dt)

config = SDQNConfig(displacement=2.0, target_height=0.0)


neighbors_layer = SignalLayerFactory(
    positions_getter=get_neighbor_positions, label="Drones Signal"
)
users_layer = SignalLayerFactory(
    positions_getter=get_user_positions, label="Users Signal"
)
frame_factory = FrameGeneratorFactory(
    geometry_factory=SquareGeometryFactory(num_cells=64, radius=1000.0),
    layer_factories=[neighbors_layer],
)

sim = SDQNTrainer(
    num_drones=num_drones,
    num_users=num_users,
    dt=dt,
    sdqn_config=config,
    frame_factory=frame_factory,
    model_path="models/sdqn_test_model.keras",
)

sim.environment.set_rectangular_boundary([-size, -size], [+size, +size])


def create_environment():
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

    sim.initialize(spacing=10.0)


# create_environment()

gui = None
gui = SDQNViewer(sim, min_fps=1.0, max_fps=1.0)
# gui = SDQNLogPolarViewer(sim, min_fps=1.0, max_fps=1.0)

for episode in range(num_episodes + 1):
    create_environment()
    # gui.reset() if gui else None

    cumulative_reward = 0.0
    episode_losses = []
    for step in range(max_steps):
        sim.update()
        fps = gui.update(force=False) if gui else np.nan

        cumulative_reward += np.mean(sim.rewards)
        if sim.sdqn_brain.wrapper.train_steps > 0:
            episode_losses.append(sim.sdqn_brain.wrapper.loss)

        print(
            (
                f"Episode: {episode}, "
                f"Step: {step + 1}, "
                f"Sim time: {sim.sim_time:.2f} s, "
                f"Real time: {sim.real_time:.2f} s, "
                f"Area cov: {sim.metrics.area_coverage*100:.2f} %, "
                f"User cov: {sim.metrics.users_coverage*100:.2f} %, "
                f"Direct conn: {sim.metrics.direct_connections*100:.2f} %, "
                f"Global conn: {sim.metrics.global_connections*100:.2f} %, "
                f"Cum reward: {cumulative_reward:.2f}, "
                + sim.sdqn_brain.wrapper.training_status_str()
            ),
            end="\r",
        )

        # if np.any(sim.dones):
        #     break

    print()
