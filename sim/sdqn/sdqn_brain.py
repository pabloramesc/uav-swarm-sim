import numpy as np
from numpy.typing import NDArray

from ..environment import Environment
from .actions import Action
from .reward_manager import RewardManager
from .sdqn_interface import SDQNInterface
from .sdqn_wrapper import SDQNWrapper


class SDQNBrain:
    def __init__(self, wrapper: SDQNWrapper, environment: Environment) -> None:
        self.wrapper = wrapper
        self.ifaces: list[SDQNInterface] = []

        self.reward_manager = RewardManager(environment)
        
        self.reset_experiences()

    @property
    def num_ifaces(self) -> int:
        return len(self.ifaces)

    def register_interface(self, iface: SDQNInterface) -> None:
        if not isinstance(iface, SDQNInterface):
            raise ValueError("iface must be a SDQNInterface instance.")
        if any(a.iface_id == iface.iface_id for a in self.ifaces):
            raise ValueError(f"Interface {iface.iface_id} already registered.")
        self.ifaces.append(iface)

    def reset_experiences(self) -> None:
        """Clear all experiences cache. Run this at the beggining of a new episode."""
        self.frames: NDArray[np.uint8] | None = None
        self.actions: NDArray[np.int32] | None = None
        self.rewards: NDArray[np.float32] | None = None
        self.dones: NDArray[np.bool_] | None = None
        self.prev_frames: NDArray[np.uint8] | None = None
        self.prev_actions: NDArray[np.int32] | None = None

    def step(
        self, drone_positions: NDArray[np.float64], user_positions: NDArray[np.float64]
    ) -> None:
        self.update_positions(drone_positions, user_positions)
        self.frames = self.generate_frames()
        self.actions = self.wrapper.act(self.frames)
        self.update_actions(self.actions)

    def train_step(
        self, drone_positions: NDArray[np.float64], user_positions: NDArray[np.float64]
    ) -> None:
        self.step(drone_positions, user_positions)

        self.rewards, self.dones = self.reward_manager.update(
            drone_positions, user_positions
        )

        if self.prev_frames is not None and self.prev_actions is not None:
            self.wrapper.add_experiences(
                frames=self.prev_frames,
                actions=self.prev_actions,
                next_frames=self.frames,  # type: ignore
                rewards=self.rewards,
                dones=self.dones,
            )
            self.wrapper.train()

        self.prev_frames = self.frames
        self.prev_actions = self.actions

    def generate_frames(self) -> np.ndarray:
        frames = np.zeros((self.num_ifaces, *self.wrapper.frame_shape), dtype=np.uint8)
        for i, iface in enumerate(self.ifaces):
            frame = iface.generate_frame()
            self.wrapper.check_frame(frame)
            frames[i] = frame
        return frames

    def update_positions(
        self, drones: NDArray[np.float64], users: NDArray[np.float64]
    ) -> None:
        self.check_positions(drones)
        self.check_positions(users)
        for i, iface in enumerate(self.ifaces):
            iface.update_positions(
                agent=drones[i], drones=np.delete(drones, i, axis=0), users=users
            )

    def update_actions(self, actions: NDArray[np.int32]) -> None:
        for i, agent in enumerate(self.ifaces):
            action = Action(actions[i])
            agent.update_action(action)

    def check_positions(self, pos: NDArray[np.float64]):
        if not isinstance(pos, np.ndarray) or pos.dtype != np.float64:
            raise ValueError("Positions must be a numpy array of type float64.")
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("Positions must be a 2D array with shape (N, 3).")
