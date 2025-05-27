import numpy as np
from .sdqn_wrapper import SDQNWrapper
from .sdqn_interface import SDQNInterface
from .actions import Action


class SDQNBrain:
    def __init__(self, wrapper: SDQNWrapper) -> None:
        self.wrapper = wrapper
        self.ifaces: list[SDQNInterface] = []

        self.last_frames: np.ndarray = None
        self.last_actions: np.ndarray = None

    @property
    def num_ifaces(self) -> int:
        return len(self.ifaces)

    def register_interface(self, iface: SDQNInterface) -> None:
        for a in self.ifaces:
            if a.iface_id == iface.iface_id:
                raise ValueError(
                    f"Interface {iface.iface_id} has already been registered."
                )
        if not isinstance(iface, SDQNInterface):
            raise ValueError("iface must be a SDQNInterface instance.")
        self.ifaces.append(iface)

    def step(self) -> None:
        if self.num_ifaces == 0:
            return

        frames = self.generate_frames()
        self.last_frames = frames

        actions = self.wrapper.act(frames)
        for i, agent in enumerate(self.ifaces):
            action = Action(actions[i])
            agent.update_action(action)
        self.last_actions = actions

        return

    def generate_frames(self) -> np.ndarray:
        frames = np.zeros((self.num_ifaces, *self.wrapper.frame_shape), dtype=np.uint8)
        for i, iface in enumerate(self.ifaces):
            frame = iface.generate_frame()
            self.wrapper.check_frame(frame)
            frames[i] = frame
        return frames

    def update_positions(self, drones: np.ndarray, users: np.ndarray) -> None:
        for i, iface in enumerate(self.ifaces):
            iface.update_positions(
                position=drones[i], drones=np.delete(drones, i, axis=0), users=users
            )
