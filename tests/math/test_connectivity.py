import unittest

import numpy as np

from sim.math.connectivity import (
    connected_clusters,
    globally_connected,
    pairwise_connectivity_matrix,
)


class ConnectivityTests(unittest.TestCase):
    def test_pairwise_matrix_is_symmetric_without_self_links(self):
        positions = np.array(
            [[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [1_000.0, 0.0, 10.0]]
        )

        matrix = pairwise_connectivity_matrix(positions)

        self.assertEqual(matrix.shape, (3, 3))
        np.testing.assert_array_equal(matrix, matrix.T)
        self.assertFalse(np.diag(matrix).any())

    def test_empty_positions_have_empty_connectivity(self):
        matrix = pairwise_connectivity_matrix(np.zeros((0, 3)))

        self.assertEqual(matrix.shape, (0, 0))
        self.assertEqual(connected_clusters(matrix), [])
        np.testing.assert_array_equal(
            globally_connected(np.zeros((0, 3))), np.array([], dtype=np.intp)
        )

    def test_largest_connected_cluster_need_not_be_first(self):
        matrix = np.array(
            [
                [False, False, False, False],
                [False, False, True, True],
                [False, True, False, True],
                [False, True, True, False],
            ]
        )
        clusters = connected_clusters(matrix)

        self.assertEqual([cluster.tolist() for cluster in clusters], [[0], [1, 2, 3]])
        # Pick a threshold that produces the same explicit topology.
        positions = np.array(
            [
                [10_000.0, 0.0, 10.0],
                [0.0, 0.0, 10.0],
                [1.0, 0.0, 10.0],
                [2.0, 0.0, 10.0],
            ]
        )
        np.testing.assert_array_equal(globally_connected(positions), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
