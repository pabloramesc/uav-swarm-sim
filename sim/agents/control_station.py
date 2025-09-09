from typing import Optional

from ..environment import Environment
from ..network.swarm_link import SwarmLink
from .agent import Agent
from .dynamics import StaticDynamics


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
        swarm_link: Optional[SwarmLink] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="gcs",
            dynamics=StaticDynamics(),
            environment=env,
        )

        self.swarm_link = swarm_link

    def initialize(self, state, time=0):
        return super().initialize(state, time)

    def update(self, dt: float = 0.01, **kwargs) -> None:
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
            self.swarm_link.update(time=self.time, position=self.dynamics.position)
