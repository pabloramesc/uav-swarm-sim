"""
Base abstract class for frame layers single-channel generators.
"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from ....environment import Environment
from ..geometry.base import FrameGeometry
from ..state import ScenarioState


class FrameLayer(ABC):
    def __init__(
        self,
        geometry: FrameGeometry,
        environment: Environment | None = None,
        label: str = "layer",
        plot_center: bool = True,
    ):
        self.geometry = geometry
        self.environment = environment
        self.label = label
        self.plot_center = plot_center
        self.position_history = deque(maxlen=10)

    def absolute_cell_ground_positions(self, state: ScenarioState) -> np.ndarray:
        """Return every frame cell as an absolute ground position."""

        positions = np.zeros((self.geometry.num_cells, 3))
        positions[:, :2] = self.geometry.flat_cell_positions + state.agent_position[:2]
        if self.environment is not None:
            positions[:, 2] = self.environment.get_elevation(positions[:, :2])
        return positions

    def relative_cell_ground_positions(self, state: ScenarioState) -> np.ndarray:
        """Return every ground cell relative to the observing agent."""

        return self.absolute_cell_ground_positions(state) - state.agent_position

    @abstractmethod
    def build_frame(self, state: ScenarioState) -> np.ndarray:
        """Override in child classes to generate layer-specific data."""
        pass

    def generate_frame(self, state: ScenarioState) -> np.ndarray:
        frame = self.build_frame(state)

        if self.plot_center:
            self.set_frame_cells(frame, positions=np.zeros(2), value=1.0, clip=True)

        return frame

    def set_frame_cells(
        self,
        frame: np.ndarray,
        positions: np.ndarray,
        value: float = 1.0,
        clip: bool = False,
    ) -> None:
        indices = self.geometry.positions_to_cell_indices(positions, clip)
        frame[indices[:, 0], indices[:, 1]] = value


class FrameLayerFactory(ABC):
    """Abstract factory for frame layers."""

    @abstractmethod
    def create(self, geo: FrameGeometry, env: Environment | None = None) -> FrameLayer:
        """Return a configured FrameLayer instance."""
        pass
