"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from .agent import Agent
from ..environment import Environment
from ..network.swarm_link import SwarmLink


class ControlStation(Agent):
    """
    Represents a control station (Ground Control Station, GCS) in the simulation environment.

    The control station is responsible for monitoring and managing other agents in the simulation.
    It does not move or perform random walks like other agents but can update its internal state.

    Attributes
    ----------
    id : int
        Unique identifier for the control station.
    """

    def __init__(
        self,
        global_id: int,
        type_id: int,
        env: Environment,
        net: SwarmLink = None,
    ):
        super().__init__(
            global_id=global_id, type_id=type_id, agent_type="gcs", env=env, net=net
        )

    def update(self, dt: float = 0.01) -> None:
        """
        Updates the internal state of the control station.

        This method can be extended to include additional logic for managing other agents.

        Parameters
        ----------
        dt : float, optional
            The time step in seconds (default is 0.01).
        """
        super().update(dt)
