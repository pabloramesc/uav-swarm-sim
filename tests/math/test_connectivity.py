import numpy as np
from multiagent_sim.math.connectivity import (
    wireless_connectivity_matrix,
    connected_clusters,
    global_connected
)

def test_connectivity_matrix_shape():
    positions = np.random.rand(10, 3) * 100
    matrix = wireless_connectivity_matrix(positions)
    assert matrix.shape == (10, 10)
    assert not np.any(np.diag(matrix))  # No self-links

def test_connected_clusters_output():
    positions = np.array([
        [0, 0, 10],
        [100, 0, 10],    # Close to 0
        [1e3, 0, 10] # Far away
    ])
    matrix = wireless_connectivity_matrix(positions)  # Very lenient
    clusters = connected_clusters(matrix)
    assert isinstance(clusters, list)
    assert len(clusters) >= 1
    total_nodes = sum(len(c) for c in clusters)
    assert total_nodes == len(positions)

def test_largest_connected_cluster():
    positions = np.array([
        [0, 0, 10],
        [100, 0, 10],
        [200, 0, 10],
        [1e3, 0, 10],
    ])
    largest = global_connected(positions)
    assert isinstance(largest, np.ndarray)
    assert len(largest) >= 1

if __name__ == "__main__":
    test_connectivity_matrix_shape()
    test_connected_clusters_output()
    test_largest_connected_cluster()