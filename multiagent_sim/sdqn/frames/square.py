import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .base import FrameBase, FrameFactory


class SquareFrame(FrameBase, ABC):
    """Abstract base class for square frames with same width and height."""

    def __init__(
        self,
        num_cells: int,
        frame_radius: float,
        channels: int = 1,
        plot_center: bool = True,
        label: str = "square_frame",
    ):
        super().__init__(
            height=num_cells, width=num_cells, channels=channels, label=label
        )

        self.num_cells = num_cells
        self.frame_radius = frame_radius
        self.cell_size = 2 * frame_radius / num_cells
        self.plot_center = plot_center

        self.cell_positions = self._calculate_cell_positions()
        self.flat_cell_positions = self.cell_positions.reshape(-1, 2)

    @abstractmethod
    def set_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def update_frame(self):
        if self.plot_center:
            self.set_center_cells(value=1.0)

    def _calculate_cell_positions(self):
        dx = np.linspace(-self.frame_radius, +self.frame_radius, self.num_cells)
        dy = np.linspace(-self.frame_radius, +self.frame_radius, self.num_cells)
        x_grid, y_grid = np.meshgrid(dx, dy)
        cell_positions = np.stack((x_grid, y_grid), axis=-1)
        return cell_positions

    def positions_to_cell_indices(self, positions: np.ndarray):
        indices = (positions - self.cell_positions[0, 0] // self.cell_size).astype(int)
        # Filter out indices that are outside the frame
        valid_mask = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < self.num_cells)
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < self.num_cells)
        )
        indices = indices[:, [1, 0]]  # Swap x and y to row, col order
        return indices[valid_mask]

    def set_cells(self, positions: np.ndarray, value: float = 1.0):
        indices = self.positions_to_cell_indices(positions)
        self.frame[indices[:, 0], indices[:, 1]] = value

    def set_center_cells(self, value: float = 1.0):
        center = self.num_cells // 2
        if self.num_cells % 2 == 0:  # Even-sized matrix
            self.frame[center - 1 : center + 1, center - 1 : center + 1] = value
        else:  # Odd-sized matrix
            self.frame[center - 1 : center + 2, center - 1 : center + 2] = value


@dataclass
class SquareFrameFactory(FrameFactory):
    label: str = "square_frame"

    def create(self):
        raise NotImplementedError
