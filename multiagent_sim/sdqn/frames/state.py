"""
Scenario state dataclass to store the current agents' positions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ScenarioState:
    agent_position: np.ndarray
    neighbor_positions: np.ndarray
    user_positions: np.ndarray
