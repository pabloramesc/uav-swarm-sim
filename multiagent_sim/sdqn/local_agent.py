from .frame_generators import FrameGenerator, SimpleFrameGenerator
from .actions import Action, ZoomableAction, action_to_displacement

import numpy as np


class LocalAgent:

    def __init__(
        self,
        agent_id: int,
        frame_generator: SimpleFrameGenerator,
    ):
        self.agent_id = agent_id
        self.frame_generator = frame_generator

        self.position = np.zeros(2)
        self.drones = np.zeros((0, 2))
        self.users = np.zeros((0, 2))
        
        self.last_frame: np.ndarray = None
        self.last_action: int = None
        self.displacement = np.zeros(2)

    def update(
        self, position: np.ndarray, drones: np.ndarray, users: np.ndarray
    ) -> None:
        self.position = position.copy()
        self.drones = drones.copy()
        self.users = users.copy()

    def generate_frame(self) -> np.ndarray:
        self.last_frame = self.frame_generator.generate_frame()
        return self.last_frame
        
    def update_action(self, action: Action) -> None:
        self.last_action = action
        
        if self.last_action == ZoomableAction.ZOOM_IN:
            new_radius = self.frame_generator.frame_radius / 2
        elif self.last_action == ZoomableAction.ZOOM_OUT:
            new_radius = self.frame_generator.frame_radius * 2
        else:
            new_radius = None
            
        if new_radius is not None:
            new_radius = np.clip(new_radius, 0.1, 10e3)
            self.frame_generator.set_frame_radius(new_radius)
            
        self.displacement = action_to_displacement(self.last_action)
