import numpy as np
from simulator.environment import Environment
from simulator.agents import Drone

class MultidroneGymDQNS:
    def __init__(
        self,
        num_drones: int,
        dt: float = 0.01,
        visible_distance: float = 100.0,
    ) -> None:
        self.num_drones = num_drones
        self.dt = dt
        self.visible_distance = visible_distance
        
        self.environment = Environment()

        self.time = 0.0
        self.step = 0
        
        self.drones = []
        for id in range(self.num_drones):
            drone = Drone(id, self.env, )
            self.drones.append(drone)
        
        self.drones_states = np.zeros((self.num_drones, 6)) # px, py, pz, vx, vy, vz
        
        
    def initialize() -> None:
        pass
    
    def update() -> None:
        pass
    
    def calculate_rewards() -> np.ndarray:
        pass