import logging
from typing import Optional

import numpy as np

from ..agents import AgentsManager, AgentsRegistry
from ..environment import Environment
from ..utils.exit_signal import register_exit_signal
from .clock import SimulationClock
from .metrics import MetricsSnapshot
from .network import NetworkManager

logger = logging.getLogger(__name__)


class MultiAgentSimulator:
    """Base class for multi-agent simulations with drones, users, and GCS agents."""

    def __init__(
        self,
        agents: AgentsManager,
        environment: Environment,
        dt: float = 0.01,
        use_network: bool = False,
    ) -> None:
        """Initializes the multi-agent simulator.

        Args:
            agents: Manager with agent registries.
            environment: Environment manager with elevation data, boundaries and obstacles.
            dem_path: Path to DEM terrain file.
            use_network: Enable network simulation.
        """
        self.agents = agents
        self.environment = environment

        self.clock = SimulationClock(dt)

        self.network = (
            NetworkManager(
                num_gcs=self.gcs.size,
                num_drones=self.drones.size,
                num_users=self.users.size,
            )
            if use_network
            else None
        )

        self.metrics: MetricsSnapshot | None = None

        # States caches [px, py, pz, vx, vy, vz]
        self.gcs_states = np.zeros((self.gcs.size, 6))
        self.user_states = np.zeros((self.users.size, 6))
        self.drone_states = np.zeros((self.drones.size, 6))

        register_exit_signal()

    @property
    def dt(self) -> float:
        return self.clock.dt

    @property
    def num_agents(self) -> int:
        return self.agents.all_agents.size

    @property
    def drones(self) -> AgentsRegistry:
        return self.agents.get_registry("drone")

    @property
    def users(self) -> AgentsRegistry:
        return self.agents.get_registry("user")

    @property
    def gcs(self) -> AgentsRegistry:
        return self.agents.get_registry("gcs")

    def reset(self, states: np.ndarray) -> None:
        """Reset simulation and set all agents to specified states.

        This method initializes each agent with a given state, resets the
        simulation clock, updates the network (if any), and refreshes internal
        state caches and metrics.

        Args:
            states: Initial states array for agents with shape (N, 6),
                where N is the number of agents.

        Raises:
            ValueError: If `states` does not have shape (num_agents, 6).
        """
        expected_shape = (self.num_agents, 6)
        if states.shape != expected_shape:
            raise ValueError(f"States array shape must be {expected_shape}.")

        self.clock.start()

        for i, agent in enumerate(self.agents.all_agents):
            agent.initialize(state=states[i, :], time=self.clock.sim_time)

        if self.network is not None:
            positions = self.agents.all_agents.get_positions_dict()
            self.network.initialize(positions)

        self._update_states_cache()
        self._update_metrics()

    def step(self, dt: Optional[float] = None, sync: bool = False) -> None:
        """Advance the simulation by one time step.

        Updates all agents, optionally synchronizes with real time, and
        refreshes internal states and metrics.

        Args:
            dt: Time step in seconds. If None, uses the default time step.
            sync: If True, synchronize simulation time with real time by
                sleeping until real time reaches simulation time. If the
                simulation is behind, no sleep occurs.
        """

        dt = self.clock.tick(dt)

        for agent in self.agents.all_agents:
            agent.update(dt)

        if self.network is not None:
            positions = self.agents.all_agents.get_positions_dict()
            self.network.update(self.clock.sim_time, positions)

        if sync:
            self.sync()

        self._update_states_cache()
        self._update_metrics()

    def sync(self) -> None:
        """Synchronize simulation time with real-world time.

        This will adjust the simulation clock and perform any necessary synchronization
        with external components (e.g., NS-3 network simulator).
        """
        self.clock.sync()
        self._sync_with_ns3()

    def _update_metrics(self) -> None:
        self.metrics = MetricsSnapshot(
            env=self.environment,
            drone_states=self.drone_states,
            user_states=self.user_states,
        )

    def _sync_with_ns3(self) -> None:
        if self.network is None:
            return
        while True:
            ns3_delta = self.clock.sim_time - self.network.ns3_time
            if ns3_delta < self.clock.sync_tolerance:
                break
            self.network.wait(timeout=ns3_delta)

    def _update_states_cache(self) -> None:
        states = self.agents.get_states()
        self.gcs_states = states["gcs"]
        self.user_states = states["user"]
        self.drone_states = states["drone"]
