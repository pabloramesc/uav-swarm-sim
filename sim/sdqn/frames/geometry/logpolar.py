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

    @property
    def xlim(self) -> tuple[float, float]:
        return (-np.pi, +np.pi)

    @property
    def ylim(self) -> tuple[float, float]:
        return (self.min_radius, self.max_radius)

    @property
    def ylabel(self) -> str:
        return "Radius (m)"

    @property
    def xlabel(self) -> str:
        return "Theta (rad)"

    @property
    def yticks(self) -> list[float]:
        return np.linspace(self.min_radius, self.max_radius, 6).tolist()

    @property
    def ytick_labels(self) -> list[str]:
        radials = np.geomspace(self.min_radius, self.max_radius, 6)
        return [f"{r:.2f}" for r in radials]

    def calculate_cell_positions(self):
        log_r_min = np.log(self.min_radius)
        log_r_max = np.log(self.max_radius)
        radial = np.exp(np.linspace(log_r_min, log_r_max, self.num_radial))
        angular = np.linspace(-np.pi, +np.pi, self.num_angular, endpoint=False)

        r_grid, theta_grid = np.meshgrid(radial, angular, indexing="ij")
        x_grid = r_grid * np.cos(theta_grid)
        y_grid = r_grid * np.sin(theta_grid)

        cell_positions = np.stack((x_grid, y_grid), axis=-1)
        return cell_positions

    def positions_to_cell_indices(
        self, positions: np.ndarray, clip: bool = False
    ) -> np.ndarray:
        pos = np.atleast_2d(positions)
        r = np.linalg.norm(pos, axis=1)
        theta = np.arctan2(pos[:, 1], pos[:, 0])

        log_r = np.log(np.clip(r, self.min_radius, self.max_radius))
        log_r_min = np.log(self.min_radius)
        log_r_max = np.log(self.max_radius)

        radial_norm = (log_r - log_r_min) / (log_r_max - log_r_min)
        radial_indices = np.floor(radial_norm * self.num_radial).astype(int)
        radial_indices = np.clip(radial_indices, 0, self.num_radial - 1)

        angular_norm = (theta + np.pi) / (2 * np.pi)
        angular_indices = np.round(angular_norm * self.num_angular).astype(int)
        angular_indices = np.mod(angular_indices, self.num_angular)

        # Filter positions out of radial bounds
        if not clip:
            valid_mask = (radial_norm >= 0.0) & (radial_norm <= 1.0)
            radial_indices = radial_indices[valid_mask]
            angular_indices = angular_indices[valid_mask]

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
