import unittest

import numpy as np

from sim.environment import (
    Environment,
    PolygonalObstacle,
    grid_positions,
    sample_positions,
)
from sim.environment.generation import reset_random_environment


class EnvironmentTests(unittest.TestCase):
    def test_instances_do_not_share_obstacle_lists(self):
        left = Environment()
        right = Environment()

        left.add_circular_obstacle((0.0, 0.0), 1.0)

        self.assertEqual(len(left.obstacles), 1)
        self.assertEqual(len(right.obstacles), 0)

    def test_grid_positions_are_compact_and_predictable(self):
        positions = grid_positions(5, origin=(10.0, 20.0), spacing=2.0, altitude=30.0)

        np.testing.assert_allclose(
            positions,
            [
                [10.0, 20.0, 30.0],
                [10.0, 22.0, 30.0],
                [10.0, 24.0, 30.0],
                [12.0, 20.0, 30.0],
                [12.0, 22.0, 30.0],
            ],
        )

    def test_random_generation_and_placement_are_seeded(self):
        first = Environment()
        second = Environment()
        reset_random_environment(first, 100.0, 3, rng=np.random.default_rng(7))
        reset_random_environment(second, 100.0, 3, rng=np.random.default_rng(7))

        first_positions = sample_positions(
            4, first, altitude=10.0, rng=np.random.default_rng(8)
        )
        second_positions = sample_positions(
            4, second, altitude=10.0, rng=np.random.default_rng(8)
        )

        np.testing.assert_allclose(first_positions, second_positions)
        self.assertTrue(first.is_inside(first_positions).all())
        self.assertFalse(first.is_collision(first_positions).any())

    def test_polygon_obstacle_edges_count_as_collisions(self):
        environment = Environment()
        environment.set_rectangular_boundary((-10.0, -10.0), (10.0, 10.0))
        environment.add_obstacle(
            PolygonalObstacle([(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)])
        )

        self.assertTrue(environment.is_collision(np.array([1.0, 0.0, 1.0])).item())


if __name__ == "__main__":
    unittest.main()
