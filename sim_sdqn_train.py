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


dt = 0.1
num_drones = 16
num_users = 20
size = 1e3
num_obstacles = 0
num_episodes = 1000
max_steps = int(5 * 60 / dt)

config = SDQNConfig(displacement=2.0, target_height=0.0)
sim = SDQNTrainer(
    num_drones=num_drones,
    num_users=num_users,
    dt=dt,
    sdqn_config=config,
    model_path="data/models/sdqn-m102.keras",
    actions_mode="basic",
    logpolar=True,
    train_mode=True,
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


logger = CSVLogger(
    filepath="logs/sdqn_m101.csv",
    columns=[
        "episode",
        "steps",
        "duration",
        "reward",
        "loss",
        "epsilon",
        "area_coverage",
        "user_coverage",
        "direct_conn",
        "global_conn",
    ],
    header_lines=[
        "Model: log-polar (binary maps), Actions: basic, Environment: 2km x 2km, ",
        f"Num drones: {num_drones}, Num users: {num_users}, Num obstacles: {num_obstacles}",
        f"Max steps: {max_steps}, Time step: {dt:.2f}",
    ],
)

# create_environment()

gui = None
# gui = SDQNViewer(sim, min_fps=1.0, max_fps=1.0)
gui = SDQNLogPolarViewer(sim, min_fps=1.0, max_fps=1.0)

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
                f"Episode: {episode}/{num_episodes}, "
                f"Step: {step + 1}/{max_steps}, "
                f"Sim time: {sim.sim_time:.2f} s, "
                f"Real time: {sim.real_time:.2f} s, "
                f"User cov: {sim.metrics.user_coverage*100:.2f} %, "
                f"Global conn: {sim.metrics.global_conn*100:.2f} %, "
                f"Cum reward: {cumulative_reward:.2f}, "
                f"Loss: {sim.sdqn_brain.wrapper.loss:.4e}, "
                f"Epsilon: {sim.sdqn_brain.wrapper.epsilon:.4f}, "
                f"Train speed: {sim.sdqn_brain.wrapper.train_speed:.2f} steps/s"
            ),
            end="\r",
        )

        # if np.any(sim.dones):
        #     break

    print(
        f"Episode: {episode}/{num_episodes}, "
        f"Steps: {step + 1}, "
        f"Sim time: {sim.sim_time:.2f} s, "
        f"Real time: {sim.real_time:.2f} s, "
        f"Area cov: {sim.metrics.area_coverage*100:.2f} %, "
        f"User cov: {sim.metrics.user_coverage*100:.2f} %, "
        f"Direct conn: {sim.metrics.direct_conn*100:.2f} %, "
        f"Global conn: {sim.metrics.global_conn*100:.2f} %, "
        f"Cum reward: {cumulative_reward:.2f}, "
        f"Loss: {sim.sdqn_brain.wrapper.loss:.4e}, "
        f"Epsilon: {sim.sdqn_brain.wrapper.epsilon:.4f}, "
        f"Train steps: {sim.sdqn_brain.wrapper.train_steps}, "
        f"Train speed: {sim.sdqn_brain.wrapper.train_speed:.2f} steps/s, "
        f"Train elapsed: {sim.sdqn_brain.wrapper.train_elapsed:.2f} s"
    )

    logger.log(
        episode=episode,
        steps=step + 1,
        duration=sim.real_time,
        reward=cumulative_reward,
        loss=np.mean(episode_losses),
        epsilon=sim.sdqn_brain.wrapper.epsilon,
        area_coverage=sim.metrics.area_coverage * 100.0,
        user_coverage=sim.metrics.user_coverage * 100.0,
        direct_conn=sim.metrics.direct_conn * 100.0,
        global_conn=sim.metrics.global_conn * 100.0,
    )
