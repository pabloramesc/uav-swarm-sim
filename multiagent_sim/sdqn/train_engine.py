import numpy as np

from .sdqn_wrapper import SDQNWrapper
from .central_agent import CentralAgent
from .reward_manager import RewardManager
from .frame_generators import SimpleFrameGenerator
from .actions import Action, ZoomableAction
from .local_agent import LocalAgent
from ..environment import Environment


class TrainEngine:

    def __init__(self, num_drones: int, env: Environment) -> None:

        dummy_frame_generator = SimpleFrameGenerator(env)
        sdqn = SDQNWrapper(
            frame_shape=dummy_frame_generator.frame_shape,
            num_actions=7,
            train_mode=True,
        )
        self.central_agent = CentralAgent(sdqn)
        self.reward_manager = RewardManager(env)

        for i in range(num_drones):
            frame_generator = SimpleFrameGenerator(env)
            agent = LocalAgent(i, frame_generator)
            self.central_agent.register_agent(i, agent)

        self.drone_positions: np.ndarray = None
        self.user_positions: np.ndarray = None
        self.prev_frames: np.ndarray = None
        self.prev_actions: np.ndarray = None

    @property
    def num_drones(self) -> int:
        return len(self.local_agents)

    def update(self, time: float, drones: np.ndarray, users: np.ndarray) -> np.ndarray:
        self.drone_positions = drones.copy()
        self.user_positions = users.copy()

        self.update_local_agents(drones, users)

        frames = self.generate_frames()
        actions = self.act(frames)

        rewards, dones = self.reward_manager.update(drones, time)
        
        self.central_agent.add_experiences(frames=self.prev_frames, next_frames=frames, actions=)
        
        return dones

    def update_local_agents(self, drones: np.ndarray, users: np.ndarray) -> None:
        if drones.shape[0] != self.num_drones:
            raise ValueError("Drone positions number do not match local agents")

        for i, agent in enumerate(self.local_agents):
            agent.update(position=drones[i], drones=np.delete(drones, i), users=users)

    def generate_frames(self) -> np.ndarray:
        frames = np.zeros((self.num_agents, *self.frame_shape))
        for i, agent in enumerate(self.local_agents):
            frames[i] = agent.generate_frame()
        return frames

    def act(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[0] != self.num_drones:
            raise ValueError("Frames number do not match local agents")

        actions = self.central_agent.act(frames)
        for i, agent in enumerate(self.local_agents):
            action = Action(actions[i])
            frames[i] = agent.update_action(action)
        return actions

   
