from .base import Dynamics

class StaticDynamics(Dynamics):
    
    def step(self, dt: float) -> None:
        return None