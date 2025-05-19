"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from multiagent_sim.core.sdqn_simulator import SDQNSimulator, SDQNConfig
from multiagent_sim.gui.sdqn_viewer import SDQNViewer

dt = 0.1
num_drones = 16
num_users = 10
size = 1e3
chnage_interval = 1000

config = SDQNConfig(displacement=10.0, target_height=10.0)
sim = SDQNSimulator(
    num_drones,
    num_users,
    dt,
    sdqn_config=config,
    model_path="data/models/sdqn-m01.keras",
    actions_mode="basic",
    train_mode=True,
)

sim.environment.set_rectangular_boundary([-size, -size], [+size, +size])

def generate_obstacles():
    sim.environment.clear_obstacles()

    for _ in range(5):
        center = np.random.uniform(-size, +size, size=(2,))
        radius = np.random.uniform(0.01 * size, 0.1 * size)
        sim.environment.add_circular_obstacle(center, radius)

    for _ in range(5):
        bottom_left = np.random.uniform(-size, +size, size=(2,))
        width_height = np.random.uniform(0.05 * size, 0.5 * size, size=(2,))
        top_right = bottom_left + width_height
        sim.environment.add_rectangular_obstacle(bottom_left, top_right)

generate_obstacles()
sim.initialize()

# gui = SDQNViewer(sim)

while True:
    sim.update()
    
    if sim.sim_step % chnage_interval == 0:
        print("Changing obstacles...")
        sim.initialize()
    
    # fps = gui.update(force=False)

    # print(f"FPS: {gui.fps:.2f}")
    print(sim.simulation_status_str())
    print(sim.training_status_str())
    print("Rewards:", " ".join(f"{r:.2f}" for r in sim.rewards))
    print()
