from typing import Literal

import numpy as np

AgentType = Literal["drone", "user", "gcs"]


class Agent:

    def __init__(self, id: int, type: AgentType):
        self.id = id
        self.type = type
        self.time = 0.0
        self.state = np.zeros(6)  # px, py, pz, vx, vy, vz

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]

    def initialize(self, state: np.ndarray) -> None:
        self._check_state(state)
        self.time = 0.0
        self.state = state

    def update(self, dt: float = 0.01) -> None:
        self.time += dt

    def _check_state(self, state: np.ndarray) -> None:
        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array")
        if state.shape != (6,):
            raise ValueError("State must be a 1D array of shape (6,)")

    def __repr__(self) -> str:
        pos = self.position
        vel = self.velocity
        pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m"
        vel_str = f"[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] m/s"
        return f"Agent(id={self.id}, type='{self.type}', position={pos_str}, velocity={vel_str})"
