import logging

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
    get_neighbor_positions,
    get_user_positions,
)
from .metrics import MetricsSnapshot
from .random_environment import reset_random_environment
from .simulator import MultiAgentSimulator


class SDQNGymEnv:

    logger = logging.getLogger("SDQNGymEnv")

    frame_factory = FrameGeneratorFactory(
        geometry_factory=SquareGeometryFactory(side_size=64, radius=1e3),
        layer_factories=[
            ObstaclesLayerFactory(label="Obstacles Map"),
            SignalLayerFactory(
                positions_getter=get_neighbor_positions,
                label="Drones Signal",
                plot_rssi=False,
            ),
            SignalLayerFactory(
                positions_getter=get_user_positions,
                label="Users Signal",
                plot_rssi=False,
            ),
        ],
    )

    def __init__(self, dt: float = 0.1, drones_speed: float = 10.0) -> None:
        self.agents = AgentsManager()
        self.environment = Environment()
        reset_random_environment(self.environment, boundary_size=1e3)

        self.sim = MultiAgentSimulator(
            agents=self.agents, environment=self.environment, dt=dt
        )
        self.rewards_manager = RewardManager(env=self.environment)

        self.step_displacement = drones_speed * dt
        self.drones_height = 10.0
        self.frame_generators = self._create_frame_generators(num_drones=1)
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

    def reset(self, num_drones: int, num_users: int) -> NDArray[np.uint8]:
        self._create_agents(num_drones, num_users)
        self.frame_generators = self._create_frame_generators(num_drones)
        reset_random_environment(
            env=self.environment,
            # num_obstacles=np.random.randint(0, 20),
            num_obstacles=0,
            boundary_size=np.random.uniform(1e3, 3e3),
        )

        self._initialize_simulator()

        frames = self._generate_drone_frames(self.drone_positions, self.user_positions)
        return frames

    def step(
        self, actions: NDArray[np.integer]
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.bool_]]:
        # Advance simulator one step
        self.sim.step()

        # Update drone positions
        displacements = self._actions_to_displacement(actions)
        for i, drone in enumerate(self.agents.drones):
            new_position = drone.state[0:2] + displacements[i]
            if not self.environment.is_collision(
                pos=new_position, check_altitude=False, check_boundary=True
            ).item():
                drone.state[0:2] = new_position
            drone.state[2] = self.drones_height

        frames = self._generate_drone_frames(self.drone_positions, self.user_positions)
        rewards, dones = self.rewards_manager.update(
            self.drone_positions, self.user_positions
        )
        return frames, rewards, dones

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
