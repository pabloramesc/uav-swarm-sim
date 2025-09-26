import logging

from sim.simulators import SDQNTrainer, GymConfig, DQNConfig

logging.basicConfig(level=logging.INFO)

gym_config = GymConfig(
    dt=1.0,
    num_drones=4,
    num_users=20,
    drones_speed=2.0,
    drones_height=10.0,
    boundary_size=1e3,
    num_obstacles=0,
)

dqn_config = DQNConfig(
    memory_size=200_000,
    min_memory=50_000,
    update_freq=10_000,
    batch_size=32,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.1,
    epsilon_decay=1e-5,
    decay_type="linear",
    n_step=3,
    per_alpha=0.6,
    per_beta=0.4,
    per_beta_annealing=1e-6,
    autosave_freq=10_000,
)

sdqn = SDQNTrainer(
    gym_config=gym_config,
    dqn_config=dqn_config,
    model_path="data/models/sdqn_model_v02.keras",
    log_path="data/logs/sdqn_model_v02.csv",
    render=True,
)

sdqn.train(
    train_freq=1,
    max_episodes=1_000_000,
    max_episode_time=None,
    max_episode_steps=1000,
    verbose=True,
    render=False,
)
