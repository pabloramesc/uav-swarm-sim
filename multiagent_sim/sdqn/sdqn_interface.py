from .frames import FrameGenerator, ScenarioState
from .actions import Action, action_to_displacement

import numpy as np


class SDQNInterface:

    def __init__(
        self,
        iface_id: int,
        frame_generator: FrameGenerator,
    ):
        self.iface_id = iface_id
        self.frame_generator = frame_generator

        self.agent = np.zeros(2)
        self.drones = np.zeros((0, 2))
        self.users = np.zeros((0, 2))

        self.frame: np.ndarray | None = None
        self.action: int | None = None
        self.direction = np.zeros(2)

    def update_positions(
        self, agent: np.ndarray, drones: np.ndarray, users: np.ndarray
    ) -> None:
        self.agent = agent
        self.drones = drones
        self.users = users

    def generate_frame(self) -> np.ndarray:
        state = ScenarioState(
            agent_position=self.agent,
            neighbor_positions=self.drones,
            user_positions=self.users,
        )
        self.frame = self.frame_generator.generate(state)
        return self.frame

    def update_action(self, action: Action) -> None:
        self.action = action
        self.direction = action_to_displacement(action)
