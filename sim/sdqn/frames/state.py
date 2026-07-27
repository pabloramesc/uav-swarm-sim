"""
Scenario state dataclass to store the current agents' positions.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class ScenarioState:
    agent_position: np.ndarray
    neighbor_positions: np.ndarray
    user_positions: np.ndarray


PositionsGetter = Callable[[ScenarioState], np.ndarray]


def get_dummy_position(state: ScenarioState) -> np.ndarray:
    return np.zeros((0, 3))


def get_agent_position(state: ScenarioState) -> np.ndarray:
    return np.atleast_2d(state.agent_position)


def get_neighbor_positions(state: ScenarioState) -> np.ndarray:
    return state.neighbor_positions


def get_user_positions(state: ScenarioState) -> np.ndarray:
    return state.user_positions
