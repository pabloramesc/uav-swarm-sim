import numpy as np

from multiagent_sim.simulators.sdqn_trainer import SDQNTrainer, SDQNConfig
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


num_drones = 16
num_users = 0
size = 1e3
num_obstacles = 0
num_episodes = 1000
max_episode_time = 5 * 60

config = SDQNConfig(displacement=2.0, target_height=0.0)


neighbors_layer = SignalLayerFactory(
    positions_getter=get_neighbor_positions, label="Drones Signal"
)
users_layer = SignalLayerFactory(
    positions_getter=get_user_positions, label="Users Signal"
)
frame_factory = FrameGeneratorFactory(
    geometry_factory=SquareGeometryFactory(side_size=64, radius=1000.0),
    layer_factories=[neighbors_layer],
)

sim = SDQNTrainer(
    num_drones=num_drones,
    num_users=num_users,
    sdqn_config=config,
    frame_factory=frame_factory,
    model_path="models/sdqn_test_model.keras",
)

sim.environment.set_rectangular_boundary([-size, -size], [+size, +size])

gui = SDQNViewer(sim, fps=1.0)
# gui = SDQNLogPolarViewer(sim, min_fps=1.0, max_fps=1.0)

for episode in range(1, num_episodes + 1):
    sim.create_random_environment(num_obstacles)
    sim.initialize()
    gui.initialize()

    while True:
        sim.update()
        gui.update(force=False)

        print("Episode:", episode)
        print(sim.simulation_status_str)
        print(sim.training_status_str)
        print()
        
        if sim.sim_time > max_episode_time:
            break

        # if np.any(sim.dones):
        #     break

    print()
