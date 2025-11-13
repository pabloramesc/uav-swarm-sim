"""
Base abstract classes for frame geometries.
"""

from abc import ABC, abstractmethod

import numpy as np


class FrameGeometry(ABC):
    """Abstract base class for frame geometries."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.cell_positions = self.calculate_cell_positions()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def num_cells(self) -> int:
        return self.height * self.width

    @property
    def flat_cell_positions(self):
        return self.cell_positions.reshape(-1, 2)

    @property
    def xlim(self) -> tuple[float, float] | None:
        return None

    @property
    def ylim(self) -> tuple[float, float] | None:
        return None

    @property
    def xlabel(self) -> str | None:
        return None

    @property
    def ylabel(self) -> str | None:
        return None

    @property
    def xticks(self) -> list[float] | None:
        return None
    
    @property
    def yticks(self) -> list[float] | None:
        return None
    
    @property
    def xtick_labels(self) -> list[str] | None:
        return None
    
    @property
    def ytick_labels(self) -> list[str] | None:
        return None
    

    @abstractmethod
    def calculate_cell_positions(self) -> np.ndarray:
        """Compute the coordinates of each cell in the frame."""
        pass

    @abstractmethod
    def positions_to_cell_indices(
        self, positions: np.ndarray, clip: bool = False
    ) -> np.ndarray:
        """Map positions to corresponding cell indices in the frame.

        Args:
            positions (np.ndarray): (N, 2) array of (x, y) positions.
            clip (bool): If True, clip indices to grid boundaries;
                otherwise discard out-of-bounds positions.

        Returns:
            np.ndarray: (row, col) cell indices.
        """
        pass


class FrameGeometryFactory(ABC):
    """Abstract base class for geometry factories."""

    @abstractmethod
    def create(self) -> FrameGeometry:
        pass
