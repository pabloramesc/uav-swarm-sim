"""
Log-polar grid (log radial + polar coordinates) frame geometry.
"""

from dataclasses import dataclass

import numpy as np

from .base import FrameGeometry, FrameGeometryFactory


class LogPolarGeometry(FrameGeometry):
    def __init__(
        self, num_radial: int, num_angular: int, min_radius: float, max_radius: float
    ):
        self.num_radial = num_radial
        self.num_angular = num_angular
        self.min_radius = min_radius
        self.max_radius = max_radius

        super().__init__(height=num_radial, width=num_angular)

    def calculate_cell_positions(self):
        log_r_min = np.log(self.min_radius)
        log_r_max = np.log(self.max_radius)
        radial = np.exp(np.linspace(log_r_min, log_r_max, self.num_radial))
        angular = np.linspace(-np.pi, +np.pi, self.num_angular, endpoint=True)

        r_grid, theta_grid = np.meshgrid(radial, angular, indexing="ij")
        x_grid = r_grid * np.cos(theta_grid)
        y_grid = r_grid * np.sin(theta_grid)

        cell_positions = np.stack((x_grid, y_grid), axis=-1)
        return cell_positions

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


@dataclass
class LogPolarGeometryFactory(FrameGeometryFactory):
    num_radial: int
    num_angular: int
    min_radius: float
    max_radius: float

    def create(self) -> LogPolarGeometry:
        return LogPolarGeometry(
            num_radial=self.num_radial,
            num_angular=self.num_angular,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
        )
