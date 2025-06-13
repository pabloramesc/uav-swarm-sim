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
        self,
        env: Environment,
        channel_shape: tuple[int, int],
        collision_distance: float = 10.0,
    ):
        self.channel_shape = channel_shape
        self.frame_shape = (*channel_shape, 3)  # 3 channels
        self.channel_names = ["Collision risk", "Drones signal", "Users coverage"]

        self.env = env
        self.collision_distance = collision_distance

        self.position = np.zeros(2)
        self.abs_drones = np.zeros((0, 2))
        self.abs_users = np.zeros((0, 2))
        self.rel_drones = np.zeros((0, 2))
        self.rel_users = np.zeros((0, 2))

        self.rel_cell_positions = np.zeros((*self.channel_shape, 2))
        self.abs_cell_positions = np.zeros((*self.channel_shape, 2))
        self.update_rel_cells_positions()

    @staticmethod
    def calculate_frame_shape(channel_shape: tuple[int, int]) -> tuple[int, int, int]:
        return (*channel_shape, 3)

    def update_positions(
        self, position: np.ndarray, drones: np.ndarray, users: np.ndarray
    ) -> None:
        self.position = position
        self.abs_drones = drones
        self.abs_users = users
        self.rel_drones = drones - position
        self.rel_users = users - position

        self.update_abs_cells_positions()

    @abstractmethod
    def generate_frame(self) -> np.ndarray:
        pass

    @abstractmethod
    def positions_to_cell_indices(self, positions: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_rel_cells_positions(self) -> None:
        pass

    def update_abs_cells_positions(self) -> None:
        self.abs_cell_positions = self.rel_cell_positions + self.position

    def generate_frame(self) -> np.ndarray:
        frame = np.zeros(self.frame_shape)
        frame[..., 0] = self.collision_risk_heatmap()
        frame[..., 1] = self.drones_coverage_map(self.rel_drones)
        frame[..., 2] = self.users_coverage_map(self.rel_users)
        return (frame * 255.0).astype(np.uint8)

    def positions_binary_map(self, positions: np.ndarray) -> np.ndarray:
        indices = self.positions_to_cell_indices(positions)
        matrix = np.zeros(self.channel_shape)
        matrix[indices[:, 0], indices[:, 1]] = 1.0
        return matrix

    def obstacles_repulsion_heatmap(self) -> np.ndarray:
        flat_abs_cells = self.abs_cell_positions.reshape(-1, 2)
        obstacles_distances = distances_to_obstacles(self.env, flat_abs_cells)
        heatmap = obstacles_distances.reshape(self.channel_shape)
        # return gaussian_decay(heatmap, sigma=self.collision_distance)
        return heatmap <= 0.0

    def drones_repulsion_heatmap(self) -> np.ndarray:
        if self.abs_drones.shape[0] == 0:
            return np.zeros(self.channel_shape)
        flat_rel_cells = self.rel_cell_positions.reshape(-1, 2)
        neighbor_distances = pairwise_cross_distances(self.rel_drones, flat_rel_cells)
        nearest_distances: np.ndarray = np.min(neighbor_distances, axis=0)
        heatmap = nearest_distances.reshape(self.channel_shape)
        # return gaussian_decay(heatmap, sigma=self.collision_distance)
        return heatmap <= 0.0

    def collision_risk_heatmap(self) -> np.ndarray:
        obstacles_heatmap = self.obstacles_repulsion_heatmap()
        # drones_heatmap = self.drones_repulsion_heatmap()
        drones_heatmap = self.positions_binary_map(self.rel_drones)
        collision_heatmap = np.maximum(obstacles_heatmap, drones_heatmap)
        return np.clip(collision_heatmap, 0.0, 1.0)

    def signal_heatmap(self, rel_positions: np.ndarray) -> np.ndarray:
        if rel_positions.shape[0] == 0:
            return np.zeros(self.channel_shape)
        flat_rel_cells = self.rel_cell_positions.reshape(-1, 2)
        rssi = signal_strength(
            rel_positions, flat_rel_cells, f=2412, n=2.4, tx_power=20, mode="max"
        ).reshape(self.channel_shape)
        quality = rssi_to_signal_quality(rssi)
        points = self.positions_binary_map(rel_positions)
        return np.clip(quality + points, 0.0, 1.0)

    def coverage_binary_map(
        self, rel_positions: np.ndarray, rssi_min: float = -80.0
    ) -> np.ndarray:
        if rel_positions.shape[0] == 0:
            return np.zeros(self.channel_shape)
        flat_rel_cells = self.rel_cell_positions.reshape(-1, 2)
        rssi = signal_strength(
            rel_positions, flat_rel_cells, f=2412, n=2.4, tx_power=20, mode="max"
        ).reshape(self.channel_shape)
        return (rssi > rssi_min).astype(np.float32)

    def drones_coverage_map(self, drone_positions: np.ndarray) -> np.ndarray:
        frame = np.zeros(self.channel_shape)
        if drone_positions.shape[0] == 0:
            return frame
        frame = self.positions_binary_map(drone_positions)
        # frame += 0.5 * self.coverage_binary_map(drone_positions, rssi_min=-80.0)
        frame += self.signal_heatmap(drone_positions)
        return np.clip(frame, 0.0, 1.0)

    def users_coverage_map(self, user_positions: np.ndarray) -> np.ndarray:
        frame = np.zeros(self.channel_shape)
        if user_positions.shape[0] == 0:
            return frame
        frame = self.positions_binary_map(user_positions)
        # frame += 0.5 * self.coverage_binary_map(np.zeros(2), rssi_min=-80.0)
        # frame += self.signal_heatmap(user_positions)
        frame += self.signal_heatmap(rel_positions=np.zeros((1, 2)))
        return np.clip(frame, 0.0, 1.0)


class GridFrameGenerator(FrameGenerator):
    def __init__(
        self,
        env: Environment,
        num_cells: int = 64,
        frame_radius: float = 100.0,
        collision_distance: float = 10.0,
    ):
        self.num_cells = num_cells
        self.frame_radius = frame_radius
        self.cell_size = self.calculate_cell_size(num_cells, frame_radius)
        super().__init__(
            env=env,
            channel_shape=(num_cells, num_cells),
            collision_distance=collision_distance,
        )

    @staticmethod
    def calculate_frame_shape(num_cells: int = 64) -> tuple[int, int, int]:
        return (num_cells, num_cells, 3)

    @staticmethod
    def calculate_cell_size(num_cells: int = 64, frame_radius: float = 100.0) -> float:
        return 2 * frame_radius / num_cells

    def update_rel_cells_positions(self) -> None:
        dx = np.linspace(-self.frame_radius, +self.frame_radius, self.num_cells)
        dy = np.linspace(-self.frame_radius, +self.frame_radius, self.num_cells)
        x_grid, y_grid = np.meshgrid(dx, dy)
        self.rel_cell_positions = np.stack((x_grid, y_grid), axis=-1)

    def positions_to_cell_indices(self, positions: np.ndarray) -> np.ndarray:
        indices = (
            (positions - self.rel_cell_positions[0, 0]) // self.cell_size
        ).astype(int)
        # Filter out indices that are outside the frame
        valid_mask = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < self.num_cells)
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < self.num_cells)
        )
        indices = indices[:, [1, 0]]  # Swap x and y to row, col order
        return indices[valid_mask]

    def set_frame_radius(self, frame_radius: float) -> None:
        if frame_radius < 0.0:
            raise ValueError("Frame radius must be positive")
        self.frame_radius = frame_radius
        self.cell_size = self.calculate_cell_size(self.num_cells, self.frame_radius)
        self.update_rel_cells_positions()
        self.update_abs_cells_positions()

    def set_center_cells(self, matrix: np.ndarray, value: float = 1.0) -> np.ndarray:
        center = self.num_cells // 2
        if self.num_cells % 2 == 0:  # Even-sized matrix
            matrix[center - 1 : center + 1, center - 1 : center + 1] = value
        else:  # Odd-sized matrix
            matrix[center - 1 : center + 2, center - 1 : center + 2] = value
        return matrix

    def generate_frame(self) -> np.ndarray:
        frame = super().generate_frame()
        frame[..., 0] = self.set_center_cells(frame[..., 0], value=255)
        frame[..., 1] = self.set_center_cells(frame[..., 1], value=255)
        frame[..., 2] = self.set_center_cells(frame[..., 2], value=255)
        return frame


class LogPolarFrameGenerator(FrameGenerator):
    def __init__(
        self,
        env: Environment,
        num_radial: int = 64,
        num_angular: int = 64,
        min_radius: float = 1e0,
        max_radius: float = 1e3,
        collision_distance: float = 10.0,
    ):
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.min_radius = min_radius
        self.max_radius = max_radius
        super().__init__(
            env=env,
            channel_shape=(num_radial, num_angular),
            collision_distance=collision_distance,
        )
        self.positional_encoding = None  # (num_radial, num_angular, 3)
        self.update_rel_cells_positions()

    @staticmethod
    def calculate_frame_shape(
        num_radial: int = 64, num_angular: int = 64
    ) -> tuple[int, int, int]:
        # 3 original + 3 positional encoding channels
        return (num_radial, num_angular, 6)

    def get_logpolar_mesh_edges(self) -> tuple[np.ndarray, np.ndarray]:
        log_r_min = np.log(self.min_radius)
        log_r_max = np.log(self.max_radius)
        r_edges = np.exp(np.linspace(log_r_min, log_r_max, self.num_radial + 1))
        theta_edges = np.linspace(-np.pi, +np.pi, self.num_angular + 1, endpoint=True)
        return r_edges, theta_edges

    def update_rel_cells_positions(self) -> None:
        log_r_min = np.log(self.min_radius)
        log_r_max = np.log(self.max_radius)
        radial = np.exp(np.linspace(log_r_min, log_r_max, self.num_radial))
        angular = np.linspace(-np.pi, +np.pi, self.num_angular, endpoint=True)

        r_grid, theta_grid = np.meshgrid(radial, angular, indexing="ij")
        px = r_grid * np.cos(theta_grid)
        py = r_grid * np.sin(theta_grid)

        self.rel_cell_positions = np.stack([px, py], axis=-1)

        # Positional encoding: r_norm, sin(theta), cos(theta)
        r_norm = (np.log(r_grid) - log_r_min) / (log_r_max - log_r_min)
        sin_theta = (np.sin(theta_grid) + 1.0) / 2.0  # Normalize to [0, 1]
        cos_theta = (np.cos(theta_grid) + 1.0) / 2.0  # Normalize to [0, 1]
        self.positional_encoding = np.stack([r_norm, sin_theta, cos_theta], axis=-1)

    def positions_to_cell_indices(self, positions: np.ndarray) -> np.ndarray:
        rel = np.atleast_2d(positions)
        r = np.linalg.norm(rel, axis=1)
        theta = np.arctan2(rel[:, 1], rel[:, 0])

        log_r = np.log(np.clip(r, self.min_radius, self.max_radius))
        log_r_min = np.log(self.min_radius)
        log_r_max = np.log(self.max_radius)

        radial_indices = (
            (log_r - log_r_min) / (log_r_max - log_r_min) * self.num_radial
        ).astype(int)
        angular_indices = ((theta + np.pi) / (2 * np.pi) * self.num_angular).astype(int)

        radial_indices = np.clip(radial_indices, 0, self.num_radial - 1)
        angular_indices = np.mod(angular_indices, self.num_angular)
        return np.stack([radial_indices, angular_indices], axis=1)

    def generate_frame(self) -> np.ndarray:
        # Get the original 3-channel frame
        frame = super().generate_frame()
        # Concatenate positional encoding channels
        frame = np.concatenate(
            [frame, (self.positional_encoding * 255.0).astype(np.uint8)], axis=-1
        )
        return frame

