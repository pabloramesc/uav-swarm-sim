"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from .agent import Agent
from ..environment import Environment
from ..network.swarm_link import SwarmLink
from ..network.network_simulator import NetworkSimulator


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
        agent_id: int,
        env: Environment,
        network_sim: NetworkSimulator = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="gcs",
            env=env,
        )

        self.swarm_link: SwarmLink = None
        if network_sim is not None:
            self.swarm_link = SwarmLink(
                agent_id=self.agent_id,
                network_sim=network_sim,
                global_bcast_interval=1.0,
            )

    def initialize(self, state, time=0):
        return super().initialize(state, time)

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

        if self.swarm_link is not None:
            self.swarm_link.update(self.time, self.state[0:3])
