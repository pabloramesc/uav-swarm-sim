import unittest

import numpy as np

from sim.environment import Environment
from sim.sdqn.rewards import RewardConfig, RewardManager


def fixed_environment() -> Environment:
    environment = Environment(obstacles=[])
    environment.set_rectangular_boundary((0.0, 0.0), (100.0, 100.0))
    return environment


class RewardManagerTests(unittest.TestCase):
    def test_empty_and_single_drone_cases(self) -> None:
        manager = RewardManager(
            fixed_environment(),
            RewardConfig(collision_dist=0.0, users_coverage="fractional"),
        )
        empty = np.zeros((0, 3), dtype=np.float64)
        users = np.zeros((0, 3), dtype=np.float64)

        rewards, dones = manager.compute_rewards(empty, users)
        self.assertEqual(rewards.shape, (0,))
        self.assertEqual(dones.shape, (0,))

        rewards, dones = manager.compute_rewards(np.array([[50.0, 50.0, 10.0]]), users)
        np.testing.assert_array_equal(rewards, np.array([0.0], dtype=np.float32))
        self.assertFalse(dones.any())
        self.assertGreater(manager.min_separation(np.array([[50.0, 50.0, 10.0]]))[0], 0)

    def test_coincident_drones_receive_configured_collision_penalty(self) -> None:
        manager = RewardManager(
            fixed_environment(),
            RewardConfig(
                collision_dist=0.0,
                users_coverage=None,
                collision_penalty=-3.5,
            ),
        )
        drones = np.array(
            [[50.0, 50.0, 10.0], [50.0, 50.0, 10.0]],
            dtype=np.float64,
        )
        rewards, dones = manager.compute_rewards(
            drones, np.zeros((0, 3), dtype=np.float64)
        )
        np.testing.assert_array_equal(rewards, np.array([-3.5, -3.5], dtype=np.float32))
        np.testing.assert_array_equal(dones, np.array([True, True]))


if __name__ == "__main__":
    unittest.main()
