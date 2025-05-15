"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from typing import Literal
from abc import ABC, abstractmethod
import numpy as np

from ..environment import Environment
from ..math.distances import distances_from_point, pairwise_cross_distances
from ..math.path_loss_model import signal_strength, rssi_to_signal_quality

from .utils import (
    VisitedCells,
    distances_to_obstacles,
    gaussian_decay,
)


class FrameGenerator(ABC):
    def __init__(
        self, env: Environment, num_channels: int, channel_shape: tuple[int, int]
    ):
        self.env = env
        self.num_channels = num_channels
        self.channel_shape = channel_shape
        self.frame_shape = (*channel_shape, num_channels)
        self.channel_names = [f"ch{i}" for i in range(num_channels)]

        self.position = np.zeros(2)
        self.drones = np.zeros((0, 2))
        self.users = np.zeros((0, 2))

    @staticmethod
    def calculate_frame_shape(
        num_channels: int, channel_shape: tuple[int, int]
    ) -> tuple[int, int, int]:
        return (*channel_shape, num_channels)

    def update(
        self, position: np.ndarray, drones: np.ndarray, users: np.ndarray
    ) -> None:
        self.position = position
        self.drones = drones
        self.users = users

    @abstractmethod
    def generate_frame(self) -> np.ndarray:
        pass


class SimpleFrameGenerator(FrameGenerator):
    def __init__(
        self,
        env: Environment,
        num_cells: int = 64,
        frame_radius: float = 100.0,
        collision_distance: float = 10.0,
    ):
        super().__init__(env, num_channels=3, channel_shape=(num_cells, num_cells))
        self.num_cells = num_cells
        self.frame_radius = frame_radius
        self.collision_distance = collision_distance

        self.cell_size = self.calculate_cell_size(num_cells, frame_radius)
        self.channel_names = ["Collision risk", "Drones signal", "Users coverage"]

        self.cell_positions = np.zeros((*self.channel_shape, 2))

    @staticmethod
    def calculate_frame_shape(num_cells: int = 64) -> tuple[int, int, int]:
        return super().calculate_frame_shape(
            num_channels=3, channel_shape=(num_cells, num_cells)
        )()
        
    @staticmethod
    def calculate_cell_size(num_cells: int = 64, frame_radius: float = 100.0) -> float:
        return 2 * frame_radius * num_cells

    def update(
        self, position: np.ndarray, drones: np.ndarray, users: np.ndarray
    ) -> None:
        super().update(position, drones, users)
        self._update_cells_positions()

    def set_frame_radius(self, frame_radius: float) -> None:
        if frame_radius < 0.0:
            raise ValueError("Frame radius must be positive")
        self.frame_radius = frame_radius
        self._update_cells_positions()

    def generate_frame(self) -> np.ndarray:
        collision_risk = self.collision_risk_heatmap()
        drones_signal = self.drones_signal_heatmap()
        users_coverage = self.users_coverage_heatmap()

        frame = np.zeros(self.frame_shape)
        frame[..., 0] = self.set_center_cells(collision_risk, value=1.0)
        frame[..., 1] = self.set_center_cells(drones_signal, value=1.0)
        frame[..., 2] = self.set_center_cells(users_coverage, value=1.0)
        return (frame * 255.0).astype(np.uint8)

    def set_center_cells(self, matrix: np.ndarray, value: float = 1.0) -> np.ndarray:
        center = self.num_cells // 2
        if self.num_cells % 2 == 0:  # Even-sized matrix
            matrix[center - 1 : center + 1, center - 1 : center + 1] = value
        else:  # Odd-sized matrix
            matrix[center - 1 : center + 2, center - 1 : center + 2] = value
        return matrix

    def _update_cells_positions(self) -> None:
        dx = np.linspace(-self.frame_radius, +self.frame_radius, self.num_cells)
        dy = np.linspace(-self.frame_radius, +self.frame_radius, self.num_cells)
        xs = dx + self.position[0]
        ys = dy + self.position[1]
        x_grid, y_grid = np.meshgrid(xs, ys)
        self.cell_positions = np.stack((x_grid, y_grid), axis=-1)

    def poisitions_to_cell_indices(self, positions: np.ndarray) -> np.ndarray:
        indices = ((positions - self.cell_positions[0, 0]) // self.cell_size).astype(
            np.int32
        )
        # Filter out indices that are outside the frame
        valid_mask = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < self.num_cells)
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < self.num_cells)
        )
        return indices[valid_mask]

    def positions_binary_map(self, positions: np.ndarray) -> np.ndarray:
        indices = self.poisitions_to_cell_indices(positions)
        matrix = np.zeros((self.num_cells, self.num_cells))
        matrix[indices[:, 1], indices[:, 0]] = 1.0
        return matrix

    def obstacles_repulsion_heatmap(self) -> np.ndarray:
        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        obstacles_distances = distances_to_obstacles(self.env, flat_cell_positions)
        heatmap = obstacles_distances.reshape(self.channel_shape)
        return gaussian_decay(heatmap, sigma=10.0)

    def drones_repulsion_heatmap(self) -> np.ndarray:
        if self.visible_neighbors.shape[0] == 0:
            return np.zeros((self.num_cells, self.num_cells))
        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        neighbor_distances = pairwise_cross_distances(self.drones, flat_cell_positions)
        nearest_distances = np.min(neighbor_distances, axis=0)
        heatmap = nearest_distances.reshape(self.channel_shape)
        return gaussian_decay(heatmap, sigma=50.0)

    def collision_risk_heatmap(self) -> np.ndarray:
        obstacles_heatmap = self.obstacles_repulsion_heatmap()
        drones_heatmap = self.drones_repulsion_heatmap()
        collision_heatmap = np.maximum(obstacles_heatmap, drones_heatmap)
        return np.clip(collision_heatmap, 0.0, 1.0)

    def drones_signal_heatmap(self) -> np.ndarray:
        if self.drones.shape[0] == 0:
            return np.zeros(self.channel_shape)

        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        rssi = signal_strength(
            self.drones, flat_cell_positions, f=2412, n=2.4, tx_power=20, mode="max"
        )
        quality = rssi_to_signal_quality(rssi)
        drones = self.positions_binary_map(self.drones)
        return np.clip(quality + drones, 0.0, 1.0)

    def users_coverage_heatmap(self) -> np.ndarray:
        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        rssi = signal_strength(
            self.position, flat_cell_positions, f=2412, n=2.4, tx_power=20, mode="max"
        )
        quality = rssi_to_signal_quality(rssi)
        users = self.positions_binary_map(self.users)
        return np.clip(quality + users, 0.0, 1.0)


class _FrameGenerator:
    """
    SDQN (Swarm Deep Q-Network)
    """

    def __init__(
        self,
        env: Environment,
        sense_radius: float = 100.0,
        num_cells: int = 100,
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
        """
        self.env = env

        self.sense_radius = sense_radius
        self.num_cells = num_cells

        self.cell_size = 2 * sense_radius / num_cells

        self.num_channels = 3
        self.channel_shape = (self.num_cells, self.num_cells)
        self.frame_shape = (*self.channel_shape, self.num_channels)

        self.time = 0.0
        self.position = np.zeros(2)
        self.drone_positions = np.zeros((0, 2))
        self.user_positions = np.zeros((0, 2))

        self.cell_positions: np.ndarray = None

        self.visited_cells = VisitedCells(cell_size=50.0)
        self.expire_time = 60.0

        self.motion_map = np.zeros((self.num_cells, self.num_cells))
        self.motion_decay = 0.9

        self.positions_history: list[np.ndarray] = []
        self.max_history = 10

    def reset(self, position: np.ndarray, neighbors: np.ndarray, time: float) -> None:
        self.position = np.copy(position)
        self.neighbors = np.copy(neighbors)
        self.time = time
        self.visited_cells.reset()
        self.motion_map = np.zeros((self.num_cells, self.num_cells))

    def update(
        self, position: np.ndarray, neighbors: np.ndarray, time: float = None
    ) -> None:
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

        if time is not None:
            self.time = time
            self.update_visited_cells()

    def update_visible_neighbors(self) -> None:
        # neighbor_distances = distances_from_point(self.position, self.neighbors)
        # is_visible = neighbor_distances < self.sense_radius
        # self.visible_neighbors = self.neighbors[is_visible]
        self.visible_neighbors = self.neighbors

    def update_cells_positions(self) -> None:
        """
        Updates the positions of the cells in the sensing grid.
        """
        dx = np.linspace(-self.sense_radius, +self.sense_radius, self.num_cells)
        dy = np.linspace(-self.sense_radius, +self.sense_radius, self.num_cells)
        xs = dx + self.position[0]
        ys = dy + self.position[1]
        x_grid, y_grid = np.meshgrid(xs, ys)
        self.cell_positions = np.stack((x_grid, y_grid), axis=-1)

    def update_visited_cells(self) -> None:
        self.visited_cells.set_cell_time(self.position, self.time)
        self.visited_cells.set_cells_time(self.neighbors, self.time)

    def update_positions_history(self) -> np.ndarray:
        all_positions = np.vstack([self.position, self.visible_neighbors])
        self.positions_history.append(all_positions)
        if len(self.positions_history) > self.max_history:
            self.positions_history.pop(0)

    def rssi_heatmap(self, units: Literal["watts", "dbm"] = "watts") -> np.ndarray:
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
        rssi = signal_strength(
            self.visible_neighbors, flat_cell_positions, f=2.4e3, mode="max"
        )
        if units == "watts":
            rssi = 10 ** (rssi / 10)  # Convert dBm to Watts

        # Reshape the signal map back to the grid shape
        heatmap = rssi.reshape(self.channel_shape)

        # Normalize the heatmap to values between 0 and 1
        heatmap: np.ndarray = heatmap - heatmap.min()
        if np.max(heatmap) > 0.0:
            heatmap /= np.max(heatmap)

        return heatmap.astype(np.float32)

    def coverage_binary_map(self, rx_sense: float = -80) -> np.ndarray:
        # Flatten the cell positions for easier processing
        flat_cell_positions = self.cell_positions.reshape(-1, 2)

        # Compute the signal strength map for the visible neighbors
        rssi = signal_strength(
            self.visible_neighbors, flat_cell_positions, f=2.4e3, mode="max"
        )

        # Reshape the signal map back to the grid shape
        rssi_heatmap = rssi.reshape(self.channel_shape)

        return (rssi_heatmap > rx_sense).astype(np.float32)

    def collision_matrix(self) -> np.ndarray:
        obstacles_matrix = self.obstacles_matrix()
        neighbors_matrix = self.neighbors_binary_map()
        collision_matrix = np.clip(obstacles_matrix + neighbors_matrix, 0.0, 1.0)
        return collision_matrix

    def neighbor_distances_heatmap(self) -> np.ndarray:
        if self.visible_neighbors.shape[0] == 0:
            return np.zeros(self.channel_shape)
        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        distances = pairwise_cross_distances(
            self.visible_neighbors, flat_cell_positions
        )
        heatmap = distances.reshape(
            self.visible_neighbors.shape[0], self.num_cells, self.num_cells
        )
        heatmap: np.ndarray = np.min(heatmap, axis=0)
        heatmap = 1.0 - gaussian_decay(heatmap, sigma=self.sense_radius)
        # heatmap = heatmap / self.sense_radius
        # heatmap = np.clip(heatmap, 0.0, 1.0)
        # heatmap = 1.0 - heatmap
        return heatmap

    def obstacles_repulsion_heatmap(self) -> np.ndarray:
        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        obstacles_distances = distances_to_obstacles(self.env, flat_cell_positions)
        obstacles_heatmap = obstacles_distances.reshape(self.channel_shape)
        return gaussian_decay(obstacles_heatmap, sigma=10.0)

    def neighbors_repulsion_heatmap(self) -> np.ndarray:
        if self.visible_neighbors.shape[0] == 0:
            return np.zeros((self.num_cells, self.num_cells))
        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        neighbor_distances = pairwise_cross_distances(
            self.visible_neighbors, flat_cell_positions
        )
        nearest_distances = np.min(neighbor_distances, axis=0)
        neighbors_heatmap = nearest_distances.reshape(self.channel_shape)
        return gaussian_decay(neighbors_heatmap, sigma=50.0)

    def collision_risk_heatmap(
        self, obstacles_heatmap: np.ndarray, neighbors_heatmap: np.ndarray
    ) -> np.ndarray:
        collision_heatmap = np.maximum(obstacles_heatmap, neighbors_heatmap)
        return np.clip(collision_heatmap, 0.0, 1.0)

    def position_reward_heatmap(
        self, obstacles_heatmap: np.ndarray, neighbors_heatmap: np.ndarray
    ) -> np.ndarray:
        distances_heatmap = self.neighbor_distances_heatmap()
        # repulsion_heatmap = obstacles_heatmap + neighbors_heatmap
        # heatmap = np.where(repulsion_heatmap < 1e-9, distances_heatmap, 0.0)
        return np.clip(distances_heatmap, 0.0, 1.0)

    def visited_cells_time_map(
        self, obstacles_heatmap: np.ndarray, neighbors_binary_map: np.ndarray
    ) -> np.ndarray:
        flat_cell_positions = self.cell_positions.reshape(-1, 2)
        last_visited_times = self.visited_cells.get_cells_time(flat_cell_positions)

        elapsed_times = (self.time - last_visited_times) / self.expire_time

        is_obstacle = obstacles_heatmap.reshape(-1) > 0.99
        elapsed_times[is_obstacle] = -1.0

        elapsed_times_map = elapsed_times.reshape(self.channel_shape)

        visited_cells_map = elapsed_times_map - neighbors_binary_map
        return np.clip((visited_cells_map + 1) / 2, 0.0, 1.0)

    def flow_map(self) -> np.ndarray:
        weight = 1.0
        flow_map = np.zeros((self.num_cells, self.num_cells))
        for pos in self.positions_history[::-1]:
            indices = self.poisitions_to_cell_indices(pos)
            cols, rows = indices[:, 0], indices[:, 1]
            flow_map[rows, cols] += weight
            weight -= 1.0 / self.max_history
        return np.clip(flow_map, 0.0, 1.0)

    def compute_motion_map(self, neighbors_binary_map: np.ndarray) -> np.ndarray:
        self.motion_map = neighbors_binary_map + self.motion_decay * self.motion_map
        return np.clip(self.motion_map, 0.0, 1.0)

    def signal_quality_heatmap(
        self, obstacles_heatmap: np.ndarray, neighbors_heatmap: np.ndarray
    ) -> np.ndarray:
        if self.visible_neighbors.shape[0] > 0:
            flat_cell_positions = self.cell_positions.reshape(-1, 2)
            rssi = signal_strength(
                self.visible_neighbors, flat_cell_positions, f=2.4e3, mode="max"
            )
            quality = rssi_to_signal_quality(rssi)
            reward = (1.0 - quality / 100.0) ** 10
        else:
            reward = np.zeros((self.num_cells, self.num_cells))

        reward_heatmap = reward.reshape(self.channel_shape)
        reward_heatmap -= obstacles_heatmap + neighbors_heatmap

        return np.clip(reward_heatmap, 0.0, 1.0)

    def set_center_cells(self, matrix: np.ndarray, value: float = 1.0) -> np.ndarray:
        """
        Set the central 2x2 (for even-sized matrices) or 3x3 (for odd-sized matrices)
        cells in the matrix to value.

        Parameters
        ----------
        matrix : np.ndarray
            The input matrix to modify.

        Returns
        -------
        np.ndarray
            The modified matrix with the central cells set to value.
        """
        center = self.num_cells // 2

        if self.num_cells % 2 == 0:  # Even-sized matrix
            matrix[center - 1 : center + 1, center - 1 : center + 1] = value
        else:  # Odd-sized matrix
            matrix[center - 1 : center + 2, center - 1 : center + 2] = value

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
        self.update_positions_history()

        obstacles_heatmap = self.obstacles_repulsion_heatmap()
        neighbors_heatmap = self.neighbors_repulsion_heatmap()
        neighbors_binary_map = self.neighbors_binary_map()
        collision_heatmap = self.collision_risk_heatmap(
            obstacles_heatmap, neighbors_heatmap
        )
        signal_heatmap = self.signal_quality_heatmap(
            obstacles_heatmap, neighbors_heatmap
        )
        position_heatmap = self.position_reward_heatmap(
            obstacles_heatmap, neighbors_heatmap
        )
        visited_cells_map = self.visited_cells_time_map(
            obstacles_heatmap, neighbors_binary_map
        )
        motion_map = self.compute_motion_map(neighbors_binary_map)

        frame = np.zeros(self.frame_shape)
        frame[..., 0] = self.set_center_cells(collision_heatmap, value=1.0)
        frame[..., 1] = self.set_center_cells(visited_cells_map, value=0.0)
        # frame[..., 1] = self.set_center_cells(signal_heatmap, value=0.0)
        # frame[..., 1] = self.set_center_cells(position_heatmap, value=0.0)
        frame[..., 2] = self.set_center_cells(motion_map, value=1.0)
        return (frame * 255.0).astype(np.uint8)
