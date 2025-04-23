"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from typing import Literal

import numpy as np

from simulator.environment import Environment
from simulator.math.distances import distances_from_point, pairwise_cross_distances
from simulator.math.path_loss_model import signal_strength
from .utils import distances_to_obstacles, gaussian_decay


class FrameGenerator:
    """
    SDQN (Swarm Deep Q-Network)
    """

    def __init__(
        self,
        env: Environment,
        sense_radius: float = 100.0,
        num_cells: int = 100,
        num_actions: int = 9,
    ) -> None:
        """
        Initialize the DQNS class.

        Parameters
        ----------
        env : Environment
            The simulation environment.
        sense_radius : float, optional
            The sensing radius of the drone (default is 100.0).
        num_cells : int, optional
            The number of cells in the sensing grid (default is 100).
        num_actions : int, optional
            The number of possible actions (default is 9).
        """
        self.env = env

        self.num_cells = num_cells
        self.sense_radius = sense_radius
        self.cell_size = 2 * sense_radius / num_cells

        self.num_actions = num_actions

        self.position = np.zeros(2)
        self.neighbors = np.zeros((0, 2))
        self.visible_neighbors = np.zeros((0, 2))

        self.cell_positions: np.ndarray = None

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
        self.position = np.copy(position)
        self.neighbors = np.copy(neighbors)

    def update_visible_neighbors(self) -> np.ndarray:
        neighbor_distances = distances_from_point(self.position, self.neighbors)
        is_visible = neighbor_distances < self.sense_radius
        self.visible_neighbors = self.neighbors[is_visible]
        return self.visible_neighbors

    def update_cells_positions(self) -> np.ndarray:
        """
        Generates and updates the positions of the cells in the sensing grid.

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
        self.cell_positions = np.stack((x_grid, y_grid), axis=-1)
        return self.cell_positions

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
        matrix = np.zeros((self.num_cells, self.num_cells), dtype=np.float32)
        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        if self.env.boundary is not None:
            is_inside = self.env.boundary.is_inside(flat_cell_positions)
            matrix += ~is_inside.reshape(self.num_cells, self.num_cells)
        for obs in self.env.obstacles:
            is_inside = obs.is_inside(flat_cell_positions)
            matrix += is_inside.reshape(self.num_cells, self.num_cells)
        return np.clip(matrix, 0.0, 1.0)  # Ensure values are binary (0.0 or 1.0)

    def signal_matrix(self, units: Literal["watts", "dbm"] = "watts") -> np.ndarray:
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
        signals = signal_strength(
            self.visible_neighbors, flat_cell_positions, f=2.4e3, mode="max"
        )
        if units == "watts":
            signals = 10 ** (signals / 10)  # Convert dBm to Watts

        # Reshape the signal map back to the grid shape
        matrix = signals.reshape(self.num_cells, self.num_cells)

        # Normalize the heatmap to values between 0 and 1
        matrix: np.ndarray = matrix - matrix.min()
        if np.max(matrix) > 0.0:
            matrix /= np.max(matrix)

        return matrix.astype(np.float32)

    def coverage_matrix(self, rx_sense: float = -80) -> np.ndarray:
        # Flatten the cell positions for easier processing
        flat_cell_positions = self.cell_positions.reshape(-1, 2)

        # Compute the signal strength map for the visible neighbors
        signals = signal_strength(
            self.visible_neighbors, flat_cell_positions, f=2.4e3, mode="max"
        )

        # Reshape the signal map back to the grid shape
        signals_map = signals.reshape(self.num_cells, self.num_cells)

        return (signals_map > rx_sense).astype(np.float32)

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
        ).astype(np.int32)

        matrix = np.zeros((self.num_cells, self.num_cells), dtype=np.float32)
        matrix[indices[:, 1], indices[:, 0]] = 1.0

        return matrix

    def collision_matrix(self) -> np.ndarray:
        obstacles_matrix = self.obstacles_matrix()
        neighbors_matrix = self.neighbors_matrix()
        collision_matrix = np.clip(obstacles_matrix + neighbors_matrix, 0.0, 1.0)
        return collision_matrix

    def distances_matrix(self) -> np.ndarray:
        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        distances = pairwise_cross_distances(
            self.visible_neighbors, flat_cell_positions
        )
        matrix = distances.reshape(
            self.visible_neighbors.shape[0], self.num_cells, self.num_cells
        )
        matrix: np.ndarray = np.min(matrix, axis=0)
        matrix = matrix / self.sense_radius
        matrix = np.clip(matrix, 0.0, 1.0)
        matrix = 1.0 - matrix
        return matrix

    def obstacles_repulsion_heatmap(self) -> np.ndarray:
        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        obstacles_distances = distances_to_obstacles(self.env, flat_cell_positions)
        obstacles_heatmap = obstacles_distances.reshape(self.num_cells, self.num_cells)
        return gaussian_decay(obstacles_heatmap, 10.0)
    
    def neighbors_repulsion_heatmap(self) -> np.ndarray:
        if self.visible_neighbors.shape[0] == 0:
            return np.zeros((self.num_cells, self.num_cells))
        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        neighbor_distances = pairwise_cross_distances(self.visible_neighbors, flat_cell_positions)
        nearest_distances = np.min(neighbor_distances, axis=0)
        neighbors_heatmap = nearest_distances.reshape(self.num_cells, self.num_cells)
        return gaussian_decay(neighbors_heatmap, 10.0)
    
    def collision_risk_heatmap(self) -> np.ndarray:
        obstacles_heatmap = self.obstacles_repulsion_heatmap()
        neighbors_heatmap = self.neighbors_repulsion_heatmap()
        collision_heatmap = obstacles_heatmap + neighbors_heatmap
        collision_heatmap = self.set_center_cells(collision_heatmap)
        return np.clip(collision_heatmap, 0.0, 1.0)
    
    def set_center_cells(self, matrix: np.ndarray) -> np.ndarray:
        """
        Set the central 2x2 (for even-sized matrices) or 3x3 (for odd-sized matrices)
        cells in the matrix to 1.0.

        Parameters
        ----------
        matrix : np.ndarray
            The input matrix to modify.

        Returns
        -------
        np.ndarray
            The modified matrix with the central cells set to 1.0.
        """
        center = self.num_cells // 2

        if self.num_cells % 2 == 0:  # Even-sized matrix
            matrix[center - 1:center + 1, center - 1:center + 1] = 1.0
        else:  # Odd-sized matrix
            matrix[center - 1:center + 2, center - 1:center + 2] = 1.0

        return matrix

    def compute_state_frame(self) -> np.ndarray:
        """
        Generate a frame combining the signal matrix and the environment
        matrix.

        The frame is a 3D array where the first channel represents the signal
        strength map from neighbors and the second channel represents the
        obstacles and bounary map.

        Returns
        -------
        np.ndarray
            A 3D array of shape (num_cells, num_cells, 2) with values scaled
            to 0-255.
        """
        self.update_cells_positions()
        self.update_visible_neighbors()
        frame = np.zeros((self.num_cells, self.num_cells, 1))
        frame[..., 0] = self.collision_risk_heatmap()
        return (frame * 255.0).astype(np.uint8)

    def calculate_target_position(self, action: int) -> np.ndarray:
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
