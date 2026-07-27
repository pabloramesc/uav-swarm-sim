import unittest

import numpy as np

from sim.agents.dynamics.random_walker import RandomWalkerDynamics


class ZeroDirectionGenerator:
    def uniform(
        self,
        low: float,
        high: float,
        size: int | None = None,
    ) -> np.ndarray | float:
        if size == 2:
            return np.zeros(2)
        return 2.0


class RandomWalkerDynamicsTests(unittest.TestCase):
    def test_seeded_generators_produce_identical_motion(self) -> None:
        first = RandomWalkerDynamics(rng=np.random.default_rng(42))
        second = RandomWalkerDynamics(rng=np.random.default_rng(42))
        first.state = np.zeros(6)
        second.state = np.zeros(6)

        for _ in range(3):
            first.step(0.1, np.zeros(3))
            second.step(0.1, np.zeros(3))

        np.testing.assert_array_equal(first.state, second.state)

    def test_zero_length_direction_has_a_finite_fallback(self) -> None:
        dynamics = RandomWalkerDynamics(rng=ZeroDirectionGenerator())
        dynamics.state = np.zeros(6)

        dynamics.step(0.1, np.zeros(3))

        self.assertTrue(np.all(np.isfinite(dynamics.state)))
        self.assertGreater(dynamics.state[3], 0.0)


if __name__ == "__main__":
    unittest.main()
