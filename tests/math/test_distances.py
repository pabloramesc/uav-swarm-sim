import unittest

import numpy as np

from sim.math.distances import distances_from_point


class DistanceTests(unittest.TestCase):
    def test_single_position_still_returns_a_vector(self) -> None:
        distances = distances_from_point(
            np.array([0.0, 0.0]),
            np.array([[3.0, 4.0]]),
        )

        np.testing.assert_array_equal(distances, np.array([5.0]))


if __name__ == "__main__":
    unittest.main()
