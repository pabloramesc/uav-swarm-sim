"""Scenario adapter that assembles a core simulator for EVSM swarms."""

from __future__ import annotations

import logging
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from ..agents.agents_manager import AgentsManager
from ..agents.control_station import ControlStation
from ..agents.drone import Drone
from ..agents.neighbor_provider import (
    RegistryNeighborProvider,
    SwarmLinkNeighborProvider,
)
from ..agents.user import User
from ..core import SimulationSnapshot, Simulator
from ..environment.environment import Environment
from ..environment.placement import grid_positions, sample_positions
from ..metrics import MetricsCalculator, MetricsSnapshot
from ..network.manager import NetworkManager
from ..network.swarm_link import SwarmLink
from .controller import EVSMConfig, EVSMPositionController
from .monitor import EVSMMonitor

logger = logging.getLogger(__name__)


class EVSMSimulator(Simulator):
    """A ready-to-run EVSM scenario built directly on :class:`Simulator`."""

    def __init__(
        self,
        environment: Environment,
        num_drones: int,
        num_users: int = 0,
        num_gcs: int = 1,
        config: EVSMConfig | None = None,
        dt: float = 0.01,
        *,
        use_network: bool = False,
        seed: int | None = None,
        metrics_area_samples: int = 1_000,
        metrics_period: float = 1.0,
    ) -> None:
        self._validate_counts(num_gcs, num_drones, num_users)
        if not np.isfinite(metrics_period) or metrics_period <= 0.0:
            raise ValueError("metrics_period must be positive and finite.")
        self.config = config or EVSMConfig()
        placement_seed, motion_seed, metrics_seed, network_seed = (
            np.random.SeedSequence(seed).spawn(4)
        )
        self._placement_rng = np.random.default_rng(placement_seed)
        self._motion_rng = np.random.default_rng(motion_seed)
        self._metrics_rng = np.random.default_rng(metrics_seed)
        self._network_rng = np.random.default_rng(network_seed)
        self.metrics_period = float(metrics_period)
        self._last_metrics_time: float | None = None

        network = (
            NetworkManager(
                num_gcs=num_gcs,
                num_drones=num_drones,
                num_users=num_users,
            )
            if use_network
            else None
        )
        agents = AgentsManager()

        # Agent IDs intentionally match the network node IDs.
        self._create_gcs(
            agents,
            environment,
            count=num_gcs,
            network=network,
        )
        self._create_drones(
            agents,
            environment,
            count=num_drones,
            network=network,
        )
        self._create_users(
            agents,
            environment,
            count=num_users,
            network=network,
        )

        super().__init__(environment, agents, dt=dt, network=network)

        self.evsm_monitor = EVSMMonitor(self.drones)
        self.metrics_calculator = MetricsCalculator(
            environment,
            area_samples=metrics_area_samples,
            rng=self._metrics_rng,
        )
        self.metrics: MetricsSnapshot | None = None

    @property
    def use_network(self) -> bool:
        return self.network is not None

    def reset(
        self,
        home: ArrayLike = (0.0, 0.0),
        spacing: float = 5.0,
        altitude: float | None = None,
    ) -> SimulationSnapshot:
        """Reset agents around ``home``.

        ``altitude`` is the drones' height above local terrain.  Users and
        control stations start on the ground.
        """

        home_xy = np.asarray(home, dtype=np.float64)
        if home_xy.shape != (2,):
            raise ValueError("home must have shape (2,).")
        if spacing < 0.0:
            raise ValueError("spacing cannot be negative.")

        drone_altitude = (
            self.config.target_altitude if altitude is None else float(altitude)
        )
        if drone_altitude < 0.0:
            raise ValueError("altitude cannot be negative.")

        gcs_states = np.zeros((self.num_gcs, 6), dtype=np.float64)
        if self.num_gcs:
            gcs_states[:, :2] = home_xy
            gcs_states[:, 2] = self.environment.get_elevation(home_xy).item()

        drone_states = np.zeros((self.num_drones, 6), dtype=np.float64)
        drone_positions = grid_positions(
            self.num_drones,
            origin=home_xy,
            spacing=spacing,
        )
        if self.num_drones:
            drone_positions[:, 2] = (
                self.environment.get_elevation(drone_positions[:, :2]) + drone_altitude
            )
            drone_states[:, :3] = drone_positions

        user_states = np.zeros((self.num_users, 6), dtype=np.float64)
        user_states[:, :3] = sample_positions(
            self.num_users,
            self.environment,
            altitude=0.0,
            rng=self._placement_rng,
        )

        snapshot = super().reset(
            {
                "gcs": gcs_states,
                "drone": drone_states,
                "user": user_states,
            }
        )
        self._refresh_derived_state(force_metrics=True)
        logger.info(
            "Reset EVSM simulation with %d drones, %d users and %d GCS.",
            self.num_drones,
            self.num_users,
            self.num_gcs,
        )
        return snapshot

    def step(
        self,
        dt: float | None = None,
        sync: bool = False,
    ) -> SimulationSnapshot:
        snapshot = super().step(dt=dt, sync=sync)
        self._refresh_derived_state()
        return snapshot

    def _refresh_derived_state(self, *, force_metrics: bool = False) -> None:
        self.evsm_monitor.update()
        metrics_due = (
            self._last_metrics_time is None
            or self.time + 1e-12 >= self._last_metrics_time + self.metrics_period
        )
        if not force_metrics and not metrics_due:
            return
        self.metrics = self.metrics_calculator.calculate(
            self.drone_states,
            self.user_states,
        )
        self._last_metrics_time = self.time

    def _create_gcs(
        self,
        agents: AgentsManager,
        environment: Environment,
        *,
        count: int,
        network: NetworkManager | None,
    ) -> None:
        for _ in range(count):
            agent_id = agents.size
            link = self._new_link(agent_id, network, position_timeout=5.0)
            agents.register_agent(
                ControlStation(
                    agent_id=agent_id,
                    env=environment,
                    swarm_link=link,
                )
            )

    def _create_drones(
        self,
        agents: AgentsManager,
        environment: Environment,
        *,
        count: int,
        network: NetworkManager | None,
    ) -> None:
        for _ in range(count):
            agent_id = agents.size
            link = self._new_link(
                agent_id,
                network,
                local_bcast_interval=0.1,
                global_bcast_interval=1.0,
                position_timeout=5.0,
            )
            if link is None:
                provider = RegistryNeighborProvider(
                    agent_id=agent_id,
                    drones_registry=agents.drones,
                    users_registry=agents.users,
                )
            else:
                provider = SwarmLinkNeighborProvider(link)

            controller = EVSMPositionController(self.config, environment)
            drone = Drone(
                agent_id=agent_id,
                env=environment,
                controller=controller,
                provider=provider,
                swarm_link=link,
            )
            drone.dynamics.mass = self.config.agent_mass
            drone.dynamics.max_acc = self.config.max_acceleration
            agents.register_agent(drone)

    def _create_users(
        self,
        agents: AgentsManager,
        environment: Environment,
        *,
        count: int,
        network: NetworkManager | None,
    ) -> None:
        for _ in range(count):
            agent_id = agents.size
            agents.register_agent(
                User(
                    agent_id=agent_id,
                    env=environment,
                    swarm_link=self._new_link(agent_id, network),
                    rng=self._motion_rng,
                )
            )

    def _new_link(
        self,
        agent_id: int,
        network: NetworkManager | None,
        **kwargs: float,
    ) -> SwarmLink | None:
        if network is None:
            return None
        return SwarmLink(
            agent_id=agent_id,
            network_sim=network.simulator,
            rng=self._network_rng,
            **kwargs,
        )

    @staticmethod
    def _validate_counts(num_gcs: int, num_drones: int, num_users: int) -> None:
        for name, value in (
            ("num_gcs", num_gcs),
            ("num_drones", num_drones),
            ("num_users", num_users),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer.")
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")
        if num_gcs + num_drones + num_users == 0:
            raise ValueError("At least one agent is required.")
