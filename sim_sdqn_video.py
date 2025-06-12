"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import os
import imageio.v2 as imageio

import numpy as np

from multiagent_sim.core.sdqn_trainer import SDQNTrainer, SDQNConfig
from multiagent_sim.gui.sdqn_viewer import SDQNViewer
from multiagent_sim.gui.sdqn_logpolar_viewer import SDQNLogPolarViewer
from multiagent_sim.utils.data_logger import DataLogger

dt = 0.1
num_drones = 16
num_users = 20
size = 1e3
num_obstacles = 5

config = SDQNConfig(target_velocity=20.0, target_height=10.0)
sim = SDQNTrainer(
    num_drones,
    num_users,
    dt,
    sdqn_config=config,
    model_path="data/models/sdqn-m3r1e2-v2.keras",
    actions_mode="basic",
    logpolar=True,
    train_mode=False,
)

sim.environment.set_rectangular_boundary([-size, -size], [+size, +size])

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

# gui = SDQNViewer(sim)
gui = SDQNLogPolarViewer(sim)

video_folder = "videos"
os.makedirs(video_folder, exist_ok=True)
video_path = os.path.join(video_folder, "sdqn_logpolar_frame.mp4")
frames = []
last_capture_time = -1.0
fps = 10.0


while sim.sim_time <= 10.0:
    sim.update()
    fps = gui.update(force=False)
    if sim.sim_time - last_capture_time >= 1 / fps:
        frame = gui.capture_frame()
        frames.append(frame)
        last_capture_time = sim.sim_time

    print(
        f"Real time: {sim.real_time:.2f} s, Sim time: {sim.sim_time:.2f} s, FPS: {gui.fps:.2f}",
        end="\r",
    )


print(f"\nTotal frames: {len(frames)}")
imageio.mimsave(video_path, frames, fps=10, format="ffmpeg")
print(f"\nVideo guardado en: {video_path}")
