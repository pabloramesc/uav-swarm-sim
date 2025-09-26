"""
Base abstract class for frame layers single-channel generators.
"""

from abc import ABC, abstractmethod
from typing import Optional
from collections import deque

import numpy as np

from sim.environment import Environment

from ..geometry.base import FrameGeometry
from ..state import ScenarioState


class FrameLayer(ABC):
    def __init__(
        self,
        geometry: FrameGeometry,
        environment: Optional[Environment] = None,
        label: str = "layer",
        plot_center: bool = True,
    ):
        self.geometry = geometry
        self.environment = environment
        self.label = label
        self.plot_center = plot_center
        self.position_history = deque(maxlen=10)

    @property
    def cell_ground_positions(self):
        ground_positions = np.zeros((self.geometry.num_cells, 3))
        ground_positions[:, 0:2] = self.geometry.flat_cell_positions
        # TODO: Add ground elevation values from environment
        return ground_positions

    @abstractmethod
    def build_frame(self, state: ScenarioState) -> np.ndarray:
        """Override in child classes to generate layer-specific data."""
        pass

    def generate_frame(self, state: ScenarioState) -> np.ndarray:
        frame = self.build_frame(state)

        frame += np.random.normal(0.0, 0.03, size=self.geometry.shape)

        absolute_positions = self.cell_ground_positions
        absolute_positions[:, 0:2] += state.agent_position[0:2]

        # Plot environment mask (boundary + obstacles)
        if self.environment is not None:
            mask = self.environment.is_collision(
                pos=absolute_positions, check_boundary=True, check_altitude=False
            )
            self.set_frame_cells(
                frame, positions=self.geometry.flat_cell_positions[mask], value=0.9
            )

        if self.plot_center:
            self.set_frame_cells(frame, positions=np.zeros(2), value=1.0)

        decay_value = 1.0
        self.position_history.append(state.agent_position.copy())
        for pos in reversed(self.position_history):
            indices = self.geometry.positions_to_cell_indices(
                (pos - state.agent_position)[0:2]
            )
            frame[indices[:, 0], indices[:, 1]] += decay_value
            decay_value -= 0.1

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
    def create(
        self, geo: FrameGeometry, env: Optional[Environment] = None
    ) -> FrameLayer:
        """Return a configured FrameLayer instance."""
        pass
