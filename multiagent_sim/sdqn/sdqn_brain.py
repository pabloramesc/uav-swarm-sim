import numpy as np
from .sdqn_wrapper import SDQNWrapper
from .sdqn_interface import SDQNInterface
from .actions import Action
from .reward_manager import RewardManager
from multiagent_sim.environment import Environment


class SDQNBrain:
    def __init__(self, wrapper: SDQNWrapper, environment: Environment) -> None:
        self.wrapper = wrapper
        self.ifaces: list[SDQNInterface] = []

        self.reward_manager = RewardManager(environment)

        self.frames: np.ndarray = None
        self.actions: np.ndarray = None
        self.rewards: np.ndarray = None
        self.dones: np.ndarray = None
        self.prev_frames: np.ndarray = None
        self.prev_actions: np.ndarray = None

    @property
    def num_ifaces(self) -> int:
        return len(self.ifaces)

    def register_interface(self, iface: SDQNInterface) -> None:
        if not isinstance(iface, SDQNInterface):
            raise ValueError("iface must be a SDQNInterface instance.")
        if any(a.iface_id == iface.iface_id for a in self.ifaces):
            raise ValueError(f"Interface {iface.iface_id} already registered.")
        self.ifaces.append(iface)

    def update(self, drones: np.ndarray, users: np.ndarray) -> None:
        self.update_positions(drones, users)

        self.rewards, self.dones = self.reward_manager.update(drones, users)
        
        self.frames = self.generate_frames()
        
        self.actions = self.wrapper.act(self.frames)
        self.update_actions(self.actions)
        
        self.wrapper.add_experiences(
            frames=self.prev_frames,
            actions=self.prev_actions,
            next_frames=self.frames,
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

    def update_positions(self, drones: np.ndarray, users: np.ndarray) -> None:
        for i, iface in enumerate(self.ifaces):
            iface.update_positions(
                agent=drones[i], drones=np.delete(drones, i, axis=0), users=users
            )
            
    def update_actions(self, actions: np.ndarray) -> None:
        for i, agent in enumerate(self.ifaces):
            action = Action(actions[i])
            agent.update_action(action)
    
