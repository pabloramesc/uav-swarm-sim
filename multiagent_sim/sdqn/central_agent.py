import numpy as np
from .sdqn_agent import SDQNAgent
from .local_agent import LocalAgent
from .actions import Action
class CentralAgent:
    def __init__(self, sdqn: SDQNAgent) -> None:
        self.sdqn = sdqn
        self.agents: dict[int, LocalAgent] = {}
        
    @property
    def num_agents(self) -> int:
        return len(self.agents)
    
    def register_agent(self, agent_id: int, agent: LocalAgent) -> None:
        if agent_id in self.agents:
            raise ValueError(f"Agent {agent_id} has already been registered.")
        if not isinstance(agent, LocalAgent):
            raise ValueError("Agent must be a LocalAgent instance.") 
        self.agents[agent_id] = agent
        
    def step(self) -> None:
        if self.num_agents == 0:
            return
        
        frames = np.zeros((self.num_agents, *self.sdqn.frame_shape))
        for i, agent in enumerate(self.agents.values()):
            frame = agent.generate_frame()
            self.sdqn.check_frame(frame)
            frames[i] = frame
            
        actions = self.sdqn.act(frames)
        for i, agent in enumerate(self.agents.values()):
            action = Action(actions[i])
            agent.update_action(action)
            
        return