"""Small, read-only views of the state maintained by EVSM controllers."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..agents.agents_registry import AgentsRegistry
from ..agents.drone import Drone
from .controller import EVSMPositionController


class EVSMMonitor:
    """Collect edge-drone and virtual-spring topology for visualization."""

    def __init__(self, drones: AgentsRegistry) -> None:
        self.registry = drones
        self.edge_mask: NDArray[np.bool_]
        self.springs_matrix: NDArray[np.bool_]
        self._resize()

    def update(self) -> None:
        if self.springs_matrix.shape != (self.registry.size, self.registry.size):
            self._resize()
        else:
            self.edge_mask.fill(False)
            self.springs_matrix.fill(False)

        for index, agent in enumerate(self.registry):
            drone = cast(Drone, agent)
            controller = drone.position_controller
            if not isinstance(controller, EVSMPositionController):
                raise TypeError(
                    f"Drone {drone.agent_id} does not use EVSMPositionController."
                )

            self.edge_mask[index] = (
                controller.evsm.sweep_angle is not None
                and controller.evsm.is_edge_robot()
            )
            self.springs_matrix[index] = self._springs_for(controller)

    def _resize(self) -> None:
        size = self.registry.size
        self.edge_mask = np.zeros(size, dtype=bool)
        self.springs_matrix = np.zeros((size, size), dtype=bool)

    def _springs_for(self, controller: EVSMPositionController) -> NDArray[np.bool_]:
        row = np.zeros(self.registry.size, dtype=bool)
        neighbor_ids = list(controller.drone_positions)
        if not neighbor_ids:
            return row

        neighbor_indices = self.registry.get_indices(neighbor_ids)
        springs = np.asarray(controller.evsm.springs_mask, dtype=bool)
        if springs.size == 0:
            # Controllers know their initial neighbors before the first
            # control update has built the EVSM mesh.
            return row
        if neighbor_indices.shape != springs.shape:
            raise RuntimeError(
                "EVSM spring mask does not match the controller neighbor set."
            )
        row[neighbor_indices] = springs
        return row
