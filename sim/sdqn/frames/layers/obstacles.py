from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim.environment.environment import Environment
from sim.sdqn.frames.state import ScenarioState

from ..geometry.base import FrameGeometry
from .base import FrameLayer, FrameLayerFactory


class ObstaclesLayer(FrameLayer):
    environment: Environment  # annotation for type checker

    def __init__(
        self,
        geometry: FrameGeometry,
        environment: Environment,
        label: str = "layer",
    ):
        super().__init__(geometry, environment, label)

    def build_frame(self, state: ScenarioState) -> NDArray[np.float32]:
        frame = np.zeros(self.geometry.shape, dtype=np.float32)

        mask = self.environment.is_collision(
            pos=self.cell_ground_positions, check_boundary=True, check_altitude=False
        )

        self.set_frame_cells(
            frame, positions=self.cell_ground_positions[mask, 0:2], value=1.0
        )

        return frame


@dataclass
class ObstaclesLayerFactory(FrameLayerFactory):
    label: str

    def create(self, geo: FrameGeometry, env: Environment | None = None) -> FrameLayer:
        if env is None:
            raise ValueError("Obstacles layer requires environment.")
        return ObstaclesLayer(geometry=geo, environment=env, label=self.label)
