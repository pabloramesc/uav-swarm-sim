from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..agents import AgentsManager, Drone, DummyNeighborProvider, User
from ..environment import Environment
from ..mobility.position_controller import DummyPositionController
from ..mobility.utils import environment_random_positions
from ..sdqn import RewardManager
from ..sdqn.actions import action_to_displacement
from ..sdqn.frames import (
    FrameGenerator,
    FrameGeneratorFactory,
    ObstaclesLayerFactory,
    ScenarioState,
    SignalLayerFactory,
    SquareGeometryFactory,
    get_agent_position,
    get_neighbor_positions,
    get_user_positions,
)
from .metrics import MetricsSnapshot
from .random_environment import reset_random_environment
from .simulator import MultiAgentSimulator


@dataclass
class GymConfig:
    dt: float = 0.1
    num_drones: int = 1
    num_users: int = 1
    drones_speed: float = 10.0
    drones_height: float = 10.0
    # boundary_size can be a float (fixed) or a tuple (min, max)
    boundary_size: Union[float, Tuple[float, float]] = 1e3
    # num_obstacles can be an int (fixed) or a tuple (min, max)
    num_obstacles: Union[int, Tuple[int, int]] = 0


class GymEnvironment:

    obstacles_map = ObstaclesLayerFactory(label="Obstacles Map")
    drones_signal = SignalLayerFactory(
        tx_positions_getter=get_neighbor_positions,
        rx_positions_getter=None,
        coverage_mode="binary",
        label="Drones Signal",
    )
    users_signal = SignalLayerFactory(
        tx_positions_getter=get_agent_position,
        rx_positions_getter=get_user_positions,
        coverage_mode="binary",
        label="Users Coverage",
    )

    frame_factory = FrameGeneratorFactory(
        geometry_factory=SquareGeometryFactory(side_size=100, radius=500),
        layer_factories=[obstacles_map, drones_signal, users_signal],
    )

    def __init__(self, config: Optional[GymConfig] = None) -> None:
        self.config = config or GymConfig()

        self.agents = AgentsManager()
        self.environment = Environment()
        self._reset_environment()

        self.sim = MultiAgentSimulator(
            agents=self.agents, environment=self.environment, dt=self.config.dt
        )
        self.rewards_manager = RewardManager(env=self.environment)

        self.step_displacement = self.config.drones_speed * self.config.dt
        self.drones_height = self.config.drones_height
        self.frame_generators = self._create_frame_generators(
            num_drones=self.config.num_drones
        )
        self.last_frames: NDArray[np.uint8] | None = None

    @property
    def sim_time(self) -> float:
        return self.sim.clock.sim_time

    @property
    def sim_step(self) -> int:
        return self.sim.clock.sim_step

    @property
    def metrics(self) -> MetricsSnapshot:
        if self.sim.metrics is None:
            raise RuntimeError("No metrics snapshot. Simulation not initiated.")
        return self.sim.metrics

    @property
    def drone_positions(self) -> NDArray[np.float64]:
        return self.sim.drone_states[:, 0:3]

    @property
    def user_positions(self) -> NDArray[np.float64]:
        return self.sim.user_states[:, 0:3]

    def reset(
        self, num_drones: int | None = None, num_users: int | None = None
    ) -> NDArray[np.uint8]:
        num_drones = num_drones or self.config.num_drones
        num_users = num_users or self.config.num_users

        self._create_agents(num_drones, num_users)
        self.frame_generators = self._create_frame_generators(num_drones)
        self._reset_environment()

        self._initialize_simulator()

        frames = self._generate_drone_frames(self.drone_positions, self.user_positions)
        return frames

    def step(
        self, actions: NDArray[np.integer]
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.bool_]]:
        # Advance simulator one step
        self.sim.step()

        # Calculate new drone positions
        displacements = self._actions_to_displacement(actions)
        new_positions = np.zeros_like(self.drone_positions)
        new_positions[:, 0:2] = (
            self.sim.drone_states[:, 0:2] + self.step_displacement * displacements
        )
        new_positions[:, 2] = self.config.drones_height

        # Compute rewards based on new positions
        rewards, dones = self.rewards_manager.compute_rewards(
            new_positions, self.user_positions
        )

        # Reset positions with collisions to previous states
        # new_positions[dones] = self.sim.drone_states[dones, 0:3]
        dones[:] = False

        # Compute frames with updated positions
        frames = self._generate_drone_frames(new_positions, self.user_positions)

        # Update drones states
        for i, drone in enumerate(self.agents.drones):
            drone.state[0:3] = new_positions[i]

        return frames, rewards, dones

    def _reset_environment(self) -> None:
        reset_random_environment(
            self.environment,
            boundary_size=self.config.boundary_size,
            num_obstacles=self.config.num_obstacles,
        )

    def _create_agents(self, num_drones: int, num_users: int) -> None:
        self.agents.clear_registries()

        for _ in range(num_drones):
            drone_id = self.agents.size + 1
            drone = Drone(
                agent_id=drone_id,
                env=self.environment,
                controller=DummyPositionController(),
                provider=DummyNeighborProvider(),
                swarm_link=None,
            )
            self.agents.register_agent(drone)

        for _ in range(num_users):
            user_id = self.agents.size + 1
            user = User(agent_id=user_id, env=self.environment, swarm_link=None)
            self.agents.register_agent(user)

    def _create_frame_generators(self, num_drones: int) -> list[FrameGenerator]:
        frame_generators: list[FrameGenerator] = []
        for _ in range(num_drones):
            frame_generator = self.frame_factory.create(env=self.environment)
            frame_generators.append(frame_generator)
        return frame_generators

    def _initialize_simulator(self) -> None:
        # Set drones random positions inside the environment boundaries
        drone_states = np.zeros((self.agents.drones.size, 6))
        drone_states[:, 0:3] = environment_random_positions(
            num_positions=self.agents.drones.size, env=self.environment
        )

        # Set users random positions
        user_states = np.zeros((self.agents.users.size, 6))
        user_states[:, 0:3] = environment_random_positions(
            num_positions=self.agents.users.size, env=self.environment
        )

        # Initialize multi-agent simulator
        states = np.vstack([drone_states, user_states])
        self.sim.reset(states)

    def _generate_drone_frames(
        self, drone_positions: NDArray[np.float64], user_positions: NDArray[np.float64]
    ) -> NDArray[np.uint8]:
        frames = []
        for i, frame_generator in enumerate(self.frame_generators):
            frame_generator = self.frame_generators[i]
            state = ScenarioState(
                agent_position=drone_positions[i],
                neighbor_positions=np.delete(drone_positions, i, axis=0),
                user_positions=user_positions,
            )
            frame = frame_generator.generate(state, dtype="uint8")
            frames.append(frame)
        self.last_frames = np.array(frames)
        return self.last_frames

    def _actions_to_displacement(
        self, actions: NDArray[np.integer]
    ) -> NDArray[np.float64]:
        actions = np.asarray(actions)
        expected_shape = (self.agents.drones.size,)
        if actions.shape != expected_shape:
            raise ValueError(f"Actions must be an array with shape {expected_shape}.")

        displacements = np.zeros((actions.size, 2), dtype=np.float64)
        for i, action in enumerate(actions):
            direction = action_to_displacement(action)
            displacements[i] = direction * self.step_displacement

        return displacements
