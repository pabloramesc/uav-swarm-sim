import numpy as np

from enum import IntEnum


class Action(IntEnum):
    NOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    ZOOM_IN = 5
    ZOOM_OUT = 6


def action_to_displacement(action: Action) -> np.ndarray:
    direction_map = {
        Action.UP: np.array([0, 1]),
        Action.DOWN: np.array([0, -1]),
        Action.LEFT: np.array([-1, 0]),
        Action.RIGHT: np.array([1, 0]),
    }
    return direction_map.get(action, np.zeros(2))
