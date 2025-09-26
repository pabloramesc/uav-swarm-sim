import logging
from typing import Optional

import numpy as np

from ..agents import AgentsManager, Drone, RegistryNeighborProvider, User
from ..environment import Environment
from ..mobility.position_controller import DummyPositionController
from ..mobility.sdqn_position_controller import SDQNConfig
from ..mobility.utils import environment_random_positions
from ..sdqn import SDQNBrain, SDQNInterface, DQNWrapper
from ..sdqn.frames import (
    FrameGeneratorFactory,
    SignalLayerFactory,
    SquareGeometryFactory,
    get_neighbor_positions,
    get_user_positions,
)
from .metrics import MetricsSnapshot
from .simulator import MultiAgentSimulator


class SDQNSimulator:

    logger = logging.getLogger("SDQNSimulator")

    def __init__(
        self,
        num_drones: int,
        num_users: int = 0,
        sdqn_config: Optional[SDQNConfig] = None,
        frame_factory: Optional[FrameGeneratorFactory] = None,
        model_path: Optional[str] = None,
        dt: float = 0.01,
    ) -> None:
        # Configuration
        self.sdqn_config = sdqn_config or SDQNConfig()
        self.frame_factory = frame_factory or self._default_frame_factory()
        self.model_path = model_path

        # Core components
        self.agents = AgentsManager()
        self.environment = Environment()
        self.sdqn_brain = self._create_sdqn_brain()

        # Agents
        self._create_drones(num_drones)
        self._create_users(num_users)

        # Simulation
        self.sim = MultiAgentSimulator(
            agents=self.agents, environment=self.environment, dt=dt, use_network=False
        )

        self.displacement = self.sdqn_config.target_velocity * self.sim.dt

    @property
    def sim_time(self) -> float:
        return self.sim.clock.sim_time

    @property
    def real_time(self) -> float:
        return self.sim.clock.real_time

    @property
    def metrics(self) -> MetricsSnapshot:
        if self.sim.metrics is None:
            raise RuntimeError("Simulator not initiated.")
        return self.sim.metrics

    def _create_sdqn_brain(self) -> SDQNBrain:
        frame_shape = self.frame_factory.create(env=None).shape
        wrapper = DQNWrapper(
            frame_shape=frame_shape,
            model_path=self.model_path,
            train_mode=False,
        )
        brain = SDQNBrain(wrapper=wrapper, environment=self.environment)
        return brain

    def _create_sdqn_interface(self, iface_id: int) -> SDQNInterface:
        frame_generator = self.frame_factory.create(env=self.environment)
        iface = SDQNInterface(iface_id, frame_generator)
        return iface

    def _create_drones(self, num_drones: int) -> None:
        for _ in range(num_drones):
            drone_id = self.agents.size + 1

            iface = self._create_sdqn_interface(iface_id=drone_id)
            self.sdqn_brain.register_interface(iface)

            provider = RegistryNeighborProvider(
                agent_id=drone_id,
                drones_registry=self.agents.get_registry("drone"),
                users_registry=self.agents.get_registry("user"),
            )

            drone = Drone(
                agent_id=drone_id,
                env=self.environment,
                controller=DummyPositionController(),
                provider=provider,
                swarm_link=None,
            )
            self.agents.register_agent(drone)
        return

    def _create_users(self, num_users: int) -> None:
        for _ in range(num_users):
            user = User(
                agent_id=self.agents.size + 1, env=self.environment, swarm_link=None
            )
            self.agents.register_agent(user)
        return

    def _default_frame_factory(self) -> FrameGeneratorFactory:
        neighbors_layer = SignalLayerFactory(
            positions_getter=get_neighbor_positions, label="Drones Signal"
        )
        users_layer = SignalLayerFactory(
            positions_getter=get_user_positions, label="Users Signal"
        )
        frame_factory = FrameGeneratorFactory(
            geometry_factory=SquareGeometryFactory(side_size=64, radius=1000.0),
            layer_factories=[neighbors_layer, users_layer],
        )
        return frame_factory

    def initialize(self) -> None:
        self.logger.info("Initializing simulation ...")

        # Set drones random positions inside the environment boundaries
        drone_states = np.zeros((self.sim.drones.size, 6))
        drone_states[:, 0:3] = environment_random_positions(
            num_positions=self.sim.drones.size, env=self.environment
        )

        # Set users random positions
        user_states = np.zeros((self.sim.users.size, 6))
        user_states[:, 0:3] = environment_random_positions(
            num_positions=self.sim.users.size, env=self.environment
        )

        # Initialize simulator
        states = np.vstack([drone_states, user_states])
        self.sim.reset(states=states)

        # Initialize SDQN Brain orchestator
        self.sdqn_brain.reset_experiences()
        self.sdqn_brain.step(
            drone_positions=drone_states[:, 0:3], user_positions=user_states[:, 0:3]
        )

        self.logger.info("✅ Initialization completed.")

    def update(self, dt: float | None = None) -> None:
        self.update_drone_positions()

        self.sim.step(dt)

        self.sdqn_brain.step(
            drone_positions=self.sim.drone_states[:, 0:2],
            user_positions=self.sim.user_states[:, 0:2],
        )

    def update_drone_positions(self) -> None:
        for drone, iface in zip(self.sim.drones, self.sdqn_brain.ifaces):
            drone.dynamics.state[0:2] += self.displacement * iface.direction
            drone.dynamics.state[2] = self.sdqn_config.target_height

    @property
    def simulation_status_str(self) -> str:
        if self.sim.metrics is None:
            raise RuntimeError("Simulator not initialized.")
        return (
            f"Real time: {self.sim.clock.real_time:.2f} s, "
            f"Sim time: {self.sim.clock.sim_time:.2f} s, "
            f"Sim steps: {self.sim.clock.sim_step}, "
            f"Area coverage: {self.sim.metrics.area_coverage*100.0:.2f} %, "
            f"Users coverage: {self.sim.metrics.users_coverage*100.0:.2f} %, "
            f"Directly connected drones: {self.sim.metrics.direct_connections*100.0:.2f} %, "
            f"Globally connected drones: {self.sim.metrics.global_connections*100.0:.2f} %"
        )