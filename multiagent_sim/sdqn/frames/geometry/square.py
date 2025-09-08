"""
Cartesian square grid (same width and height) frame geometry.
"""

from dataclasses import dataclass

import numpy as np

from .base import FrameGeometry, FrameGeometryFactory


class SquareGeometry(FrameGeometry):
    def __init__(self, side_size: int, radius: float):
        self.side_size = side_size
        self.radius = radius
        self.cell_size = 2 * radius / side_size

        super().__init__(height=side_size, width=side_size)

    def calculate_cell_positions(self):
        dx = (
            np.linspace(-self.radius, self.radius - self.cell_size, self.side_size)
            + self.cell_size / 2
        )
        x_grid, y_grid = np.meshgrid(dx, dx)  # Same spacing for X and Y axes

        cell_positions = np.stack((x_grid, y_grid), axis=-1)
        return cell_positions

    def positions_to_cell_indices(self, positions: np.ndarray, clip: bool = False):
        pos_2d = np.atleast_2d(positions)
        indices = (pos_2d - self.cell_positions[0, 0]) / self.cell_size
        indices = np.floor(indices).astype(int)

        # Clip indices to valid range
        if clip:
            indices = np.clip(indices, 0, self.side_size - 1)

        # Filter out indices that are outside the frame
        else:
            mask = (
                (indices[:, 0] >= 0)
                & (indices[:, 0] < self.side_size)
                & (indices[:, 1] >= 0)
                & (indices[:, 1] < self.side_size)
            )
            indices = indices[mask]

        indices = indices[:, [1, 0]]  # Swap x and y to row, col order
        return indices


@dataclass
class SquareGeometryFactory(FrameGeometryFactory):
    side_size: int
    radius: float

    def create(self) -> SquareGeometry:
        return SquareGeometry(side_size=self.side_size, radius=self.radius)
