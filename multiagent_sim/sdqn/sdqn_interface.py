from .frames import FrameBase
from .actions import Action, action_to_displacement

import numpy as np


class SDQNInterface:

    def __init__(
        self,
        iface_id: int,
        frame_generator: FrameBase,
    ):
        self.iface_id = iface_id
        self.frame_generator = frame_generator

        self.position = np.zeros(2)
        self.drones = np.zeros((0, 2))
        self.users = np.zeros((0, 2))

        self.frame: np.ndarray = None
        self.action: int = None
        self.direction = np.zeros(2)

    def update_positions(
        self, position: np.ndarray, drones: np.ndarray, users: np.ndarray
    ) -> None:
        self.position = position
        self.drones = drones
        self.users = users
        self.frame_generator.set_data(agent=position, tx_positions=drones, users=users)

    def generate_frame(self) -> np.ndarray:
        self.frame = self.frame_generator.generate(update=True)
        return self.frame

    def update_action(self, action: Action) -> None:
        self.action = action
        self.direction = action_to_displacement(action)
