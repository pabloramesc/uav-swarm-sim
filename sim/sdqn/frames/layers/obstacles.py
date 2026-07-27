from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ....environment.environment import Environment
from ..geometry.base import FrameGeometry
from ..state import ScenarioState
from .base import FrameLayer, FrameLayerFactory


class ObstaclesLayer(FrameLayer):
    environment: Environment  # annotation for type checker

    def __init__(
        self,
        geometry: FrameGeometry,
        environment: Environment,
        label: str = "obstacles layer",
        plot_center: bool = True,
    ):
        super().__init__(geometry, environment, label, plot_center)

    def build_frame(self, state: ScenarioState) -> NDArray[np.float32]:
        frame = np.zeros(self.geometry.shape, dtype=np.float32)

        absolute_positions = self.absolute_cell_ground_positions(state)

        # Plot environment mask (boundary + obstacles)
        env_mask = self.environment.is_collision(
            pos=absolute_positions, check_boundary=True, check_altitude=False
        )
        self.set_frame_cells(
            frame, positions=self.geometry.flat_cell_positions[env_mask], value=1.0
        )

        # Plot drones positions
        drones_pos = state.neighbor_positions - state.agent_position
        self.set_frame_cells(frame, positions=drones_pos[:, 0:2], value=1.0, clip=False)

        return frame


@dataclass
class ObstaclesLayerFactory(FrameLayerFactory):
    label: str = "obstacles layer"
    plot_center: bool = True

    def create(self, geo: FrameGeometry, env: Environment | None = None) -> FrameLayer:
        if env is None:
            raise ValueError("Obstacles layer requires environment.")
        return ObstaclesLayer(
            geometry=geo,
            environment=env,
            label=self.label,
            plot_center=self.plot_center,
        )
