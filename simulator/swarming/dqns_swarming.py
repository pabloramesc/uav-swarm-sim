"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from ..environment import Environment
from ..math.distances import calculate_relative_distances
from ..math.path_loss_model import calculate_signal_strength


class DQNS:
    """
    DQNS (Deep Q-Learning Swarming)
    """

    def __init__(self, env: Environment, num_cells: int, sense_radius: float) -> None:
        self.env = env

        self.num_cells = num_cells
        self.sense_radius = sense_radius
        self.cell_size = 2 * sense_radius / num_cells

        self.position = np.zeros(2)
        self.visible_neighbors = np.zeros((0, 2))

        self.neighbor_distances = np.zeros((0,))
        self.neighbor_relative_positions = np.zeros((0, 2))
        self.cell_positions = np.zeros((self.num_cells, self.num_cells, 2))

    def update(self, position: np.ndarray, neighbors: np.ndarray) -> None:
        self.position = position
        self.cell_positions = self.get_matrix_cells_positions()

        self.neighbor_distances = calculate_relative_distances(position, neighbors)
        is_visible = self.neighbor_distances < self.sense_radius
        self.visible_neighbors = neighbors[is_visible]
        self.neighbor_relative_positions = self.visible_neighbors - self.position[None, :]

    def get_matrix_cells_positions(self) -> np.ndarray:
        """
        Generates the positions of the cells in the sensing grid.

        Returns
        -------
        np.ndarray
            An array of shape (num_cells, num_cells, 2) containing the positions of each cell.
        """
        dx = np.linspace(-self.sense_radius, +self.sense_radius, self.num_cells)
        dy = np.linspace(-self.sense_radius, +self.sense_radius, self.num_cells)
        xs = dx + self.position[0]
        ys = dy + self.position[1]
        x_grid, y_grid = np.meshgrid(xs, ys)
        return np.stack((x_grid, y_grid), axis=-1)

    def get_environment_matrix(self) -> np.ndarray:
        """
        Generates a binary matrix indicating whether each cell is inside an obstacle boundary.

        Returns
        -------
        np.ndarray
            A binary matrix of shape (num_cells, num_cells) with 1.0 for cells inside obstacles and 0.0 otherwise.
        """
        matrix = np.zeros((self.num_cells, self.num_cells))
        if self.env.boundary is not None:
            is_inside = np.array(
                [
                    self.env.boundary.is_inside(pos)
                    for pos in self.cell_positions.reshape(-1, 2)
                ]
            )
            matrix += ~is_inside.reshape(self.num_cells, self.num_cells)
        for obs in self.env.obstacles:
            is_inside = np.array(
                [obs.is_inside(pos) for pos in self.cell_positions.reshape(-1, 2)]
            )
            matrix += is_inside.reshape(self.num_cells, self.num_cells)
        return np.clip(matrix, 0, 1)  # Ensure values are binary (0.0 or 1.0)

    def get_signal_matrix(self) -> np.ndarray:
        """
        Generates a heatmap matrix using the signal strength map where each drone position contributes to the signal.

        Returns
        -------
        np.ndarray
            A heatmap matrix of shape (num_cells, num_cells) with values normalized between 0 and 1.
        """
        # Flatten the cell positions for easier processing
        flat_cell_pos = self.cell_positions.reshape(-1, 2)

        # Compute the signal strength map for the visible neighbors
        signal_map = calculate_signal_strength(self.visible_neighbors, flat_cell_pos)

        # Reshape the signal map back to the grid shape
        heatmap = signal_map.reshape(self.num_cells, self.num_cells)

        # Normalize the heatmap to values between 0 and 1
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0.0 else heatmap

        return heatmap

    def get_neighbor_matrix(self) -> np.ndarray:
        indices = ((self.visible_neighbors - self.cell_positions[0, 0]) // self.cell_size).astype(int)
        matrix = np.zeros((self.num_cells, self.num_cells))
        matrix[indices[:, 1], indices[:, 0]] = 1.0
        return matrix
    
    def build_keras_model(self):
        pass
