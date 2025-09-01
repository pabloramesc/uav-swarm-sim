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
        self.shape = (height, width)
        self.cell_positions = self.calculate_cell_positions()

    @property
    def flat_cell_positions(self):
        return self.cell_positions.reshape(-1, 2)

    @abstractmethod
    def calculate_cell_positions(self) -> np.ndarray:
        """Compute the coordinates of each cell in the frame."""
        pass

    @abstractmethod
    def positions_to_cell_indices(self) -> np.ndarray:
        """Map positions to corresponding cell indices in the frame."""
        pass
    

class FrameGeometryFactory(ABC):
    """Abstract base class for geometry factories."""
    
    @abstractmethod
    def create(self) -> FrameGeometry:
        pass