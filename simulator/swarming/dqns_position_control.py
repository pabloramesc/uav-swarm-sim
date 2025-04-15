from dataclasses import dataclass
import numpy as np
from .base_position_control import PositionController, PositionControllerConfig
from ..environment import Environment
from .dqns_swarming import DQNS
from .altitude_control import AltitudeController
from .horizontal_position_control import HorizontalPositionController


@dataclass
class DQNSConfig(PositionControllerConfig):
    visible_distance: float = 100.0 # in meters
    obstacle_distance: float = 10.0  # in meters
    agent_mass: float = 1.0  # simple equivalence between force and acceleration
    max_acceleration: float = 10.0  # 1 g aprox. 9.81 m/s^2
    target_velocity: float = 15.0  # between 5-25 m/S
    target_altitude: float = 100.0  # in meters


class DQNSPostionController(PositionController):
    def __init__(self, config: DQNSConfig, env: Environment) -> None:
        super().__init__(config, env)
        
        self.dqns = DQNS(env=self.env, )
        self.altitude_hold = AltitudeController()
        self.position_controller = HorizontalPositionController()

    def update(
        self,
        state: np.ndarray,
        neighbor_states: np.ndarray,
        neighbor_ids: np.ndarray = None,
        time: float = None,
    ) -> np.ndarray:
        super().update(state, neighbor_states, neighbor_ids, time)
