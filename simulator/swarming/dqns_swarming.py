"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from ..environment import Environment
from ..math.distances import relative_distances
from ..math.path_loss_model import signal_strength


class DQNS:
    """
    DQNS (Deep Q-Learning Swarming)
    """

    def __init__(
        self,
        env: Environment,
        num_cells: int = 100,
        sense_radius: float = 100.0,
        num_actions: int = 9,
    ) -> None:
        """
        Initialize the DQNS class.

        Parameters
        ----------
        env : Environment
            The simulation environment.
        num_cells : int, optional
            The number of cells in the sensing grid (default is 100).
        sense_radius : float, optional
            The sensing radius of the drone (default is 100.0).
        num_actions : int, optional
            The number of possible actions (default is 9).
        """
        self.env = env

        self.num_cells = num_cells
        self.sense_radius = sense_radius
        self.cell_size = 2 * sense_radius / num_cells

        self.num_actions = num_actions

        self.position = np.zeros(2)
        self.visible_neighbors = np.zeros((0, 2))
        self.frame = np.zeros((self.num_cells, self.num_cells, 2))
        self.cell_positions = np.zeros((self.num_cells, self.num_cells, 2))

    def update(self, position: np.ndarray, neighbors: np.ndarray) -> None:
        """
        Update the drone's position, visible neighbors, and frame.

        Parameters
        ----------
        position : np.ndarray
            A (2,) array with the current horizontal position of the drone.
        neighbors : np.ndarray
            A (N, 2) array with the horizontal positions of the neighbors.
        """
        self.position = position
        self.cell_positions = self.cells_positions()

        neighbor_distances = relative_distances(position, neighbors)
        is_visible = neighbor_distances < self.sense_radius
        self.visible_neighbors = neighbors[is_visible]

        self.frame = self.state_frame()

    def cells_positions(self) -> np.ndarray:
        """
        Generates the positions of the cells in the sensing grid.

        Returns
        -------
        np.ndarray
            An array of shape (num_cells, num_cells, 2) containing the
            positions of each cell.
        """
        dx = np.linspace(-self.sense_radius, +self.sense_radius, self.num_cells)
        dy = np.linspace(-self.sense_radius, +self.sense_radius, self.num_cells)
        xs = dx + self.position[0]
        ys = dy + self.position[1]
        x_grid, y_grid = np.meshgrid(xs, ys)
        return np.stack((x_grid, y_grid), axis=-1)

    def obstacles_matrix(self) -> np.ndarray:
        """
        Generates a binary matrix indicating whether each cell is inside an
        obstacle boundary.

        Returns
        -------
        np.ndarray
            A binary matrix of shape (num_cells, num_cells) with 1.0 for cells
            inside obstacles and 0.0 otherwise.
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

    def signal_matrix(self) -> np.ndarray:
        """
        Generate a heatmap matrix using the signal strength map.

        Each drone position contributes to the signal strength in the matrix.

        Returns
        -------
        np.ndarray
            A heatmap matrix of shape (num_cells, num_cells) with values
            normalized between 0 and 1.
        """
        # Flatten the cell positions for easier processing
        flat_cell_positions = self.cell_positions.reshape(-1, 2)

        # Compute the signal strength map for the visible neighbors
        signal_map = signal_strength(self.visible_neighbors, flat_cell_positions)
        signal_map = 10 ** (signal_map / 10)  # Convert dBm to Watts

        # Reshape the signal map back to the grid shape
        heatmap = signal_map.reshape(self.num_cells, self.num_cells)

        # Normalize the heatmap to values between 0 and 1
        if np.max(heatmap) > 0.0:
            heatmap /= np.max(heatmap)

        return heatmap

    def neighbors_matrix(self) -> np.ndarray:
        """
        Generate a binary matrix indicating the presence of visible neighbors
        in each cell.

        Returns
        -------
        np.ndarray
            A binary matrix of shape (num_cells, num_cells) with 1.0 for cells
            containing neighbors and 0.0 otherwise.
        """
        indices = (
            (self.visible_neighbors - self.cell_positions[0, 0]) // self.cell_size
        ).astype(int)

        matrix = np.zeros((self.num_cells, self.num_cells))
        matrix[indices[:, 1], indices[:, 0]] = 1.0

        return matrix

    def state_frame(self) -> np.ndarray:
        """
        Generate a frame combining the signal matrix and the environment
        matrix.

        The frame is a 3D array where the first channel represents the signal
        matrix and the second channel represents the environment matrix.

        Returns
        -------
        np.ndarray
            A 3D array of shape (num_cells, num_cells, 2) with values scaled
            to 0-255.
        """
        neighbors_matrix = self.signal_matrix()
        obstacles_matrix = self.obstacles_matrix()

        frame = np.stack((neighbors_matrix, obstacles_matrix), axis=-1)
        return (frame * 255.0).astype(np.uint8)

    def target_position(self, action: int) -> np.ndarray:
        """
        Calculate the target position based on the given action.

        Parameters
        ----------
        action : int
            The action index. Action 0 corresponds to staying in the current
            position.

        Returns
        -------
        np.ndarray
            The target position as a 2D coordinate.

        Raises
        ------
        ValueError
            If the action is less than 0.
        """
        if action < 0:
            raise ValueError("Action must be equal or greater than 0.")

        if action == 0:
            return self.position

        num_quads = self.num_actions - 1
        angle = (
            2 * np.pi * (action - 1) / num_quads
        )  # Divide the quadrant into equal angles
        delta_position = self.cell_size * np.array([np.cos(angle), np.sin(angle)])

        return self.position + delta_position
