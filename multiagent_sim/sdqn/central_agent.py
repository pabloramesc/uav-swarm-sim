import numpy as np
from .sdqn_wrapper import SDQNWrapper
from .local_agent import LocalAgent
from .actions import Action


class CentralAgent:
    def __init__(self, wrapper: SDQNWrapper) -> None:
        self.sdqn = wrapper
        self.agents: list[LocalAgent] = []
        
        self.last_frames: np.ndarray = None
        self.last_actions: np.ndarray = None

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def register_agent(self, agent: LocalAgent) -> None:
        for a in self.agents:
            if a.agent_id == agent.agent_id:
                raise ValueError(f"Agent {agent.agent_id} has already been registered.")
        if not isinstance(agent, LocalAgent):
            raise ValueError("Agent must be a LocalAgent instance.")
        self.agents.append(agent)

    def step(self) -> None:
        if self.num_agents == 0:
            return

        frames = self.generate_frames()
        self.last_frames = frames

        actions = self.sdqn.act(frames)
        for i, agent in enumerate(self.agents):
            action = Action(actions[i])
            agent.update_action(action)
        self.last_actions = actions

        return
    
    def generate_frames(self) -> np.ndarray:
        frames = np.zeros((self.num_agents, *self.sdqn.frame_shape), dtype=np.uint8)
        for i, agent in enumerate(self.agents):
            frame = agent.generate_frame()
            self.sdqn.check_frame(frame)
            frames[i] = frame
        return frames
