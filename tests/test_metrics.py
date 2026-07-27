import unittest

import numpy as np

from sim.environment import Environment
from sim.metrics import MetricsCalculator


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.environment = Environment()
        self.environment.set_rectangular_boundary((0.0, 0.0), (100.0, 100.0))

    def test_empty_state_is_valid(self):
        metrics = MetricsCalculator(self.environment).calculate(
            np.zeros((0, 6)), np.zeros((0, 6))
        )

        self.assertEqual(metrics.area_coverage, 0.0)
        self.assertEqual(metrics.users_coverage, 0.0)
        self.assertEqual(metrics.direct_connections, 0.0)
        self.assertEqual(metrics.global_connections, 0.0)
        self.assertEqual(metrics.links_matrix.shape, (0, 0))

    def test_metrics_are_reproducible_with_seeded_sampling(self):
        drones = np.zeros((1, 6))
        drones[0, :3] = (50.0, 50.0, 10.0)
        users = np.zeros((1, 6))
        users[0, :3] = (55.0, 50.0, 0.0)

        first = MetricsCalculator(
            self.environment, area_samples=20, rng=np.random.default_rng(2)
        ).calculate(drones, users)
        second = MetricsCalculator(
            self.environment, area_samples=20, rng=np.random.default_rng(2)
        ).calculate(drones, users)

        self.assertEqual(first.area_coverage, second.area_coverage)
        self.assertEqual(first.users_coverage, 1.0)
        self.assertEqual(first.direct_connections, 0.0)
        self.assertEqual(first.global_connections, 1.0)

        with self.assertRaises(ValueError):
            first.covered_users[0] = 99


if __name__ == "__main__":
    unittest.main()
