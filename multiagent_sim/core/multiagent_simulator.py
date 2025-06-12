"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time
from abc import ABC, abstractmethod

import numpy as np

from ..agents import Agent, AgentsRegistry, ControlStation, User, Drone
from ..environment import Environment
from ..utils.exit_signal import register_exit_signal
from ..utils.logger import create_logger
from .metrics_generator import MetricsGenerator
from .network_manager import NetworkManager


class MultiAgentSimulator(ABC):

    SYNC_TOLERANCE = 0.1

    def __init__(
        self,
        num_drones: int,
        num_users: int = 0,
        num_gcs: int = 1,
        dt: float = 0.01,
        dem_path: str = None,
        use_network: bool = False,
        **kwargs,
    ) -> None:
        self.num_drones = num_drones
        self.num_users = num_users
        self.num_gcs = num_gcs
        self.dt = dt
        self.environment = Environment(dem_path)

        if use_network:
            self.network = NetworkManager(num_gcs, num_drones, num_users)
            self.netsim = self.network.netsim
        else:
            self.network = None
            self.netsim = None

        self.gcs = AgentsRegistry()
        self.drones = AgentsRegistry()
        self.users = AgentsRegistry()
        
        self.agents = AgentsRegistry()
        self._create_agents(**kwargs)

        self.metrics = MetricsGenerator(env=self.environment, netsim=self.netsim)

        self.logger = create_logger(name="MultiAgentSimulator", level="INFO")

        self.init_time: float = None
        self.sim_time = 0.0
        self.sim_step = 0
        
        self.drone_states = np.zeros((num_drones, 6)) # px, py, pz, vx, vy, vz
        self.user_states = np.zeros((num_users, 6))
        self.gcs_states = np.zeros((num_gcs, 6))

        register_exit_signal()

    @property
    def real_time(self) -> float:
        """
        Returns the real time elapsed since the simulation started.
        """
        return time.time() - self.init_time if self.init_time else 0.0
    
    def _create_agents(self, **kwargs) -> None:
        self._create_gcs()
        self._create_drones(**kwargs)
        self._create_users()
    
    def _create_gcs(self) -> None:
        for _ in range(self.num_gcs):
            gcs = ControlStation(
                agent_id=len(self.agents),
                environment=self.environment,
                network_sim=self.netsim,
            )
            self.gcs.register(gcs)
            self.agents.register(gcs)
        
    def _create_users(self) -> None:
        for _ in range(self.num_users):
            user = User(
                agent_id=len(self.agents),
                environment=self.environment,
                network_sim=self.netsim,
            )
            self.users.register(user)
            self.agents.register(user)

    def _create_drones(self, **kwargs) -> None:
        for _ in range(self.num_drones):
            drone = self._create_drone(**kwargs)
            self.drones.register(drone)
            self.agents.register(drone)
    
    @abstractmethod
    def _create_drone(self, **kwargs) -> Drone:
        pass
            
    @abstractmethod
    def initialize(self, *args, **kwargs) -> None:
        if self.network is not None:
            positions = self.agents.get_positions_dict()
            self.network.initialize(positions)

        self.init_time = time.time()
        self.sim_time = 0.0
        self.sim_step = 0
        
        self._update_states_cache()

    @abstractmethod
    def update(self, dt: float = None, **kwargs) -> None:
        dt = dt if dt is not None else self.dt
        self.sim_time += dt
        self.sim_step += 1

        if self.network is not None:
            positions = self.agents.get_positions_dict()
            self.network.update(self.sim_time, positions)

        for agent in self.agents:
            agent.update(dt)
        
        self._update_states_cache()
        
        self.metrics.update(
            drone_states=self.drone_states,
            user_states=self.user_states,
        )

    def _sync_to_real_time(self) -> None:
        """
        Synchronizes the simulation time with real time and NS-3 time.
        """
        real_delta = self.sim_time - self.real_time
        if real_delta > self.SYNC_TOLERANCE:
            time.sleep(real_delta)

        if self.network is not None:
            self._sync_to_ns3_time()

    def _sync_to_ns3_time(self) -> None:
        while True:
            ns3_delta = self.sim_time - self.network.ns3_time
            if ns3_delta < self.SYNC_TOLERANCE:
                break
            self.network.wait(timeout=ns3_delta)

    def _update_states_cache(self) -> None:
        self.gcs_states = self.gcs.get_states_array()
        self.user_states = self.users.get_states_array()
        self.drone_states = self.drones.get_states_array()