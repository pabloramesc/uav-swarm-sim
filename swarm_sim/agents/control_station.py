"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from simulator.agents.agent import Agent


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

    def __init__(self, id: int) -> None:
        """
        Initializes the control station with a unique ID.

        Parameters
        ----------
        id : int
            Unique identifier for the control station.
        """
        super().__init__(id=id, type="gcs")

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