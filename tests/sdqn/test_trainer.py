import unittest

import numpy as np

from sim.sdqn.actions import Action
from sim.sdqn.environment import SDQNEnvironmentConfig
from sim.sdqn.trainer import SDQNTrainer

from .test_environment import fixed_environment, small_frame_factory


class FakePolicy:
    def __init__(self) -> None:
        self.transitions = []
        self.save_count = 0

    def act(self, frames: np.ndarray) -> np.ndarray:
        return np.full(frames.shape[0], Action.NOP, dtype=np.int32)

    def add_experiences(self, *transition: np.ndarray) -> None:
        self.transitions.append(transition)

    def train(self) -> dict[str, float]:
        return {}

    def save(self) -> None:
        self.save_count += 1


class SDQNTrainerTests(unittest.TestCase):
    def test_frame_factory_configures_created_environment(self) -> None:
        trainer = SDQNTrainer(
            environment_config=SDQNEnvironmentConfig(num_users=0),
            frame_factory=small_frame_factory(),
            policy=FakePolicy(),
        )
        self.addCleanup(trainer.close)

        self.assertEqual(trainer.environment.frame_shape, (5, 5, 1))

    def make_trainer(self) -> SDQNTrainer:
        from sim.sdqn.environment import SDQNEnvironment

        environment = SDQNEnvironment(
            SDQNEnvironmentConfig(
                dt=0.25,
                num_drones=1,
                num_users=0,
                drones_speed=1.0,
            ),
            environment=fixed_environment(),
            frame_factory=small_frame_factory(),
        )
        return SDQNTrainer(environment=environment, policy=FakePolicy())

    def test_step_and_time_limits_are_independent(self) -> None:
        trainer = self.make_trainer()
        trainer.reset(seed=1)
        self.assertFalse(trainer.step(max_steps=10, max_time=0.5))
        self.assertTrue(trainer.step(max_steps=10, max_time=0.5))

        trainer.reset(seed=1)
        self.assertFalse(trainer.step(max_steps=2, max_time=100.0))
        self.assertTrue(trainer.step(max_steps=2, max_time=100.0))

    def test_collision_ends_the_shared_episode(self) -> None:
        trainer = self.make_trainer()
        trainer.reset(
            options={
                "drone_states": np.array(
                    [[0.0, 50.0, 10.0, 0.0, 0.0, 0.0]],
                    dtype=np.float64,
                )
            }
        )

        self.assertTrue(trainer.step(max_steps=10))

    def test_train_saves_final_policy(self) -> None:
        trainer = self.make_trainer()
        policy = trainer.dqn

        trainer.train(
            max_episodes=1,
            max_episode_steps=1,
            verbose=False,
        )

        self.assertEqual(policy.save_count, 1)


if __name__ == "__main__":
    unittest.main()
