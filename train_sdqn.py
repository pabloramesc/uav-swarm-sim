import logging

from sim.simulators import SDQNTrainer, GymConfig, DQNConfig
from sim.sdqn.rewards import RewardConfig

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "sdqn_model_v22"

gym_config = GymConfig(
    dt=1.0,
    num_drones=2,
    num_users=25,
    drones_speed=2.0,
    drones_height=10.0,
    boundary_size=2e3,
    num_obstacles=0,
    reward_config=RewardConfig(
        collision_dist=1.0,
        users_coverage="difference",
        weight_users_coverage=1.0,
        collision_penalty=-1.0,
    ),
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
    per_beta_annealing=0.0,
    autosave_freq=10_000,
    learning_rate=1e-4,
)

sdqn = SDQNTrainer(
    gym_config=gym_config,
    dqn_config=dqn_config,
    model_path=f"data/models/{MODEL_NAME}.keras",
    log_path=f"data/logs/{MODEL_NAME}.csv",
    render=True,
)

sdqn.train(
    train_freq=1,
    max_episodes=1_000_000,
    max_episode_time=None,
    max_episode_steps=1000,
    verbose=True,
    render=True,
)
