import numpy as np

from ..agents import AgentsRegistry, Drone
from ..mobility.evsm_position_controller import EVSMPositionController

from typing import cast


class EVSMMonitor:
    def __init__(self, drones: AgentsRegistry):
        self.registry = drones
        self.edge_mask = np.zeros(drones.size, dtype=bool)
        self.springs_matrix = np.zeros(
            (drones.size, drones.size), dtype=bool
        )

    def update(self):
        for i, agent in enumerate(self.registry):
            drone = cast(Drone, agent)
            controller = drone.position_controller

            if controller is None:
                raise Exception(f"Drone {drone.agent_id} has no position controller")

            if not isinstance(controller, EVSMPositionController):
                raise Exception(f"Drone {drone.agent_id} position controller is not EVSM")

            self.edge_mask[i] = controller.evsm.is_edge_robot()
            self.springs_matrix[i] = self._drone_springs_mask(controller)

    def _drone_springs_mask(self, controller: EVSMPositionController) -> np.ndarray:
        drone_springs = np.zeros(self.registry.size, dtype=bool)

        neighbor_ids = np.array(list(controller.drone_positions.keys()), dtype=np.intp)
        neighbor_indices = self.registry.get_indices(neighbor_ids) # type: ignore

        if neighbor_indices.size == 0:
            return drone_springs

        springs_mask = controller.evsm.springs_mask
        if neighbor_indices.shape != springs_mask.shape:
            raise ValueError("Springs mask shape do not match neighbor indices")

        drone_springs[neighbor_indices] = springs_mask
        return drone_springs
