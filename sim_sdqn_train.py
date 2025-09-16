import logging

from sim.simulators import SDQNTrainer

logging.basicConfig(level=logging.INFO)

sdqn = SDQNTrainer(
    num_drones=16,
    num_users=32,
    model_path="models/sdqn_test_model.keras",
    render=True,
)

sdqn.train(
    train_freq=4,
    max_episodes=1_000_000,
    max_episode_time=60.0 * 5,
    verbose=True,
    render=False,
)
