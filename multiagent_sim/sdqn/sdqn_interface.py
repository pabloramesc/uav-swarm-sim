from .frame_generators import FrameGenerator, SimpleFrameGenerator
from .actions import Action, action_to_displacement

import numpy as np


class SDQNInterface:

    def __init__(
        self,
        iface_id: int,
        frame_generator: SimpleFrameGenerator,
    ):
        self.iface_id = iface_id
        self.frame_generator = frame_generator

        self.position = np.zeros(2)
        self.drones = np.zeros((0, 2))
        self.users = np.zeros((0, 2))
        
        self.last_frame: np.ndarray = None
        self.last_action: int = None
        self.direction = np.zeros(2)

    def update(
        self, position: np.ndarray, drones: np.ndarray, users: np.ndarray
    ) -> None:
        self.position = position
        self.drones = drones
        self.users = users
        self.frame_generator.update(position, drones, users)

    def generate_frame(self) -> np.ndarray:
        frame = self.frame_generator.generate_frame()
        self.last_frame = frame
        return frame
        
    def update_action(self, action: Action) -> None:
        if action == Action.ZOOM_IN:
            new_radius = self.frame_generator.frame_radius / 1.1
        elif action == Action.ZOOM_OUT:
            new_radius = self.frame_generator.frame_radius * 1.1
        else:
            new_radius = None
            
        if new_radius is not None:
            new_radius = np.clip(new_radius, 10.0, 10e3)
            self.frame_generator.set_frame_radius(new_radius)
            
        self.direction = action_to_displacement(action)
        self.last_action = action
