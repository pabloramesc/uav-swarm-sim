import numpy as np

from ..agents import (
    ControlStation,
    Drone,
    RegistryNeighborProvider,
    SwarmLinkNeighborProvider,
    User,
)
from ..environment import Environment
from ..network import SwarmLink
from ..utils.exit_signal import register_exit_signal
from ..utils.logger import create_logger
from .clock import SimulationClock
from .metrics import MetricsSnapshot
from .network import NetworkManager
from .agents import AgentsManager


class MultiAgentSimulator:
    """Base class for multi-agent simulations with drones, users, and GCS agents."""

    def __init__(
        self,
        num_drones: int,
        num_users: int = 0,
        num_gcs: int = 1,
        dt: float = 0.01,
        dem_path: str = None,
        use_network: bool = False,
    ) -> None:
        """Initializes the multi-agent simulator.

        Args:
            num_drones: Number of drone agents.
            num_users: Number of user agents.
            num_gcs: Number of control stations..
            dt: Default simulation time step in seconds.
            dem_path: Path to DEM terrain file.
            use_network: Enable network simulation.
        """
        self.environment = Environment(dem_path)
        self.network = (
            NetworkManager(num_gcs, num_drones, num_users) if use_network else None
        )
        self.clock = SimulationClock(dt)

        self.agents = AgentsManager()
        self._create_agents(num_gcs=num_gcs, num_users=num_users, num_drones=num_drones)

        self.metrics: MetricsSnapshot = None
        self.logger = create_logger(name="MultiAgentSimulator", level="INFO")

        # States caches [px, py, pz, vx, vy, vz]
        self.gcs_states = np.zeros((num_gcs, 6))
        self.user_states = np.zeros((num_users, 6))
        self.drone_states = np.zeros((num_drones, 6))

        register_exit_signal()

    def _create_agents(self, num_gcs: int, num_users: int, num_drones: int) -> None:
        """Create and register all agents."""
        for _ in range(num_gcs):
            gcs = self.create_gcs()
            self.agents.register_agent(gcs)

        for _ in range(num_users):
            user = self.create_user()
            self.agents.register_agent(user)

        for _ in range(num_drones):
            drone = self.create_drone()
            self.agents.register_agent(drone)

    def create_gcs(self) -> ControlStation:
        """Create a ground control station (GCS) agent."""
        return ControlStation(
            agent_id=self.agents.size,
            env=self.environment,
            netsim=self.network.netsim if self.network else None,
        )

    def create_user(self) -> User:
        """Create an user agent."""
        return User(
            agent_id=self.agents.size,
            env=self.environment,
            netsim=self.network.netsim if self.network else None,
        )

    def create_drone(self) -> Drone:
        """Create a drone (UAV) agent."""
        agent_id = self.agents.id

        if self.network:
            swarm_link = SwarmLink(
                agent_id=agent_id,
                network_sim=self.network.netsim,
                local_bcast_interval=0.1,
                global_bcast_interval=1.0,
                position_timeout=5.0,
            )
            provider = SwarmLinkNeighborProvider(swarm_link=swarm_link)

        else:
            swarm_link = None
            provider = RegistryNeighborProvider(
                agent_id=agent_id,
                drones_registry=self.agents.get_registry("drone"),
                users_registry=self.agents.get_registry("user"),
            )

        return Drone(
            agent_id=agent_id,
            env=self.environment,
            controller=None,
            provider=provider,
            swarm_link=swarm_link,
        )

    def initialize(self, states: np.ndarray) -> None:
        """Initializes simulation and all agents.
        
        Args:
            states: Initial states array for agents with shape (N, 6).
        """
        expected_shape = (self.agents.size, 6)
        if states.shape != expected_shape:
            raise ValueError(f"States array shape must be {expected_shape}.")
        
        self.clock.start()

        for i, agent in enumerate(self.agents.all_agents):
            agent.initialize(states[i, :], time=self.clock.sim_time)

        if self.network is not None:
            positions = self.agents.all_agents.get_positions_dict()
            self.network.initialize(positions)

        self._update_states_cache()
        self._update_metrics()

    def update(self, dt: float = None) -> None:
        """Advances the simulation by one time step.

        Args:
            dt: Time step in seconds. If None, use default dt.
        """
        
        dt = self.clock.tick(dt)

        for agent in self.agents.all_agents:
            agent.update(dt)

        if self.network is not None:
            positions = self.agents.all_agents.get_positions_dict()
            self.network.update(self.sim_time, positions)

        self._update_states_cache()
        self._update_metrics()

    def sync(self) -> None:
        self.clock.sync()
        if self.network:
            self._sync_with_ns3()

    def _update_metrics(self) -> None:
        self.metrics = MetricsSnapshot(
            env=self.environment,
            drone_states=self.drone_states,
            user_states=self.user_states,
        )

    def _sync_with_ns3(self) -> None:
        while True:
            ns3_delta = self.sim_time - self.network.ns3_time
            if ns3_delta < self.clock.sync_tolerance:
                break
            self.network.wait(timeout=ns3_delta)

    def _update_states_cache(self) -> None:
        states = self.agents.get_states()
        self.gcs_states = states["gcs"]
        self.user_states = states["users"]
        self.drone_states = states["drones"]
