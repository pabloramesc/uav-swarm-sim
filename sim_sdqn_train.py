"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from multiagent_sim.core.sdqn_trainer import SDQNTrainer, SDQNConfig
from multiagent_sim.gui.sdqn_viewer import SDQNViewer
from multiagent_sim.gui.sdqn_logpolar_viewer import SDQNLogPolarViewer

import os


class TrainingLogger:
    def __init__(
        self, filepath, config, num_drones, num_users, num_obstacles, episode_steps
    ):
        self.filepath = filepath
        with open(filepath, "w") as f:
            f.write(
                f"# num_drones={num_drones}, num_users={num_users}, num_obstacles={num_obstacles}, episode_steps={episode_steps}\n"
            )
            f.write(f"# {config}\n")
            f.write(
                "episode,cumulative_reward,mean_loss,mean_area_coverage,mean_user_coverage,mean_direct_conn,mean_global_conn\n"
            )

    def log_episode(
        self,
        episode,
        cumulative_reward,
        mean_loss,
        mean_area_coverage,
        mean_user_coverage,
        mean_direct_conn,
        mean_global_conn,
    ):
        with open(self.filepath, "a") as f:
            f.write(
                f"{episode},{cumulative_reward:.4f},{mean_loss:.6f},"
                f"{mean_area_coverage:.6f},{mean_user_coverage:.6f},"
                f"{mean_direct_conn:.6f},{mean_global_conn:.6f}\n"
            )


dt = 0.1
num_drones = 16
num_users = 20
size = 1e3
num_obstacles = 0
num_episodes = 10_000
episode_steps = 1_000

config = SDQNConfig(target_velocity=20.0, target_height=0.0)
sim = SDQNTrainer(
    num_drones,
    num_users,
    dt,
    sdqn_config=config,
    model_path="data/models/sdqn-m1r1e1-1k.keras",
    actions_mode="basic",
    train_mode=True,
    logpolar=False,
)

sim.environment.set_rectangular_boundary([-size, -size], [+size, +size])


def create_environment():
    global gui
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

    sim.initialize(spacing=50.0)


logger = TrainingLogger(
    "sdqn_train_m1r1e1_1k.csv",
    config,
    num_drones,
    num_users,
    num_obstacles,
    episode_steps,
)

for episode in range(num_episodes + 1):
    create_environment()
    gui = None
    # gui = SDQNViewer(sim)
    # gui = SDQNLogPolarViewer(sim)

    cumulative_reward = 0.0
    episode_losses = []
    episode_area_coverages = []
    episode_user_coverages = []
    episode_direct_connections = []
    episode_global_connections = []
    for step in range(episode_steps):
        sim.update()
        fps = gui.update(force=False) if gui else np.nan

        if sim.sdqn_brain.wrapper.train_steps > 1:
            cumulative_reward += np.mean(sim.rewards)
            episode_losses.append(sim.sdqn_brain.wrapper.loss)
            episode_area_coverages.append(sim.metrics.area_coverage)
            episode_user_coverages.append(sim.metrics.users_coverage)
            episode_direct_connections.append(sim.metrics.direct_conn)
            episode_global_connections.append(sim.metrics.global_conn)

        print(
            f"Step: {step + 1}/{episode_steps}, "
            f"Real time: {sim.real_time:.2f} s, "
            f"Sim time: {sim.sim_time:.2f} s, "
            f"Cumulative reward: {cumulative_reward:.2f}, "
            f"Loss: {sim.sdqn_brain.wrapper.loss:.4e}, "
            f"Epsilon: {sim.sdqn_brain.wrapper.epsilon:.4f}, "
            f"Train steps: {sim.sdqn_brain.wrapper.train_steps}",
            end="\r",
        )

    print(f"\nEpisode {episode}/{num_episodes}")
    print(f"- Train metrics: {sim.training_status_str()}")
    print(f"- Cumulative Reward: {cumulative_reward:.2f}")
    print(f"- Losses: {np.mean(episode_losses):.4f}")
    print(f"- Area Coverage: {np.mean(episode_area_coverages):.4f}")
    print(f"- User Coverage: {np.mean(episode_user_coverages):.4f}")
    print(f"- Direct Connections: {np.mean(episode_direct_connections):.4f}")
    print(f"- Global Connections: {np.mean(episode_global_connections):.4f}")

    logger.log_episode(
        episode,
        cumulative_reward,
        np.mean(episode_losses),
        np.mean(episode_area_coverages),
        np.mean(episode_user_coverages),
        np.mean(episode_direct_connections),
        np.mean(episode_global_connections),
    )
