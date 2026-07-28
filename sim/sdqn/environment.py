"""Vectorized SDQN environment built directly on the simulation core."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..agents import AgentsManager, Drone, DummyNeighborProvider, User
from ..core import Simulator
from ..environment import Environment
from ..environment.generation import FloatRange, IntRange, reset_random_environment
from ..environment.placement import sample_positions
from ..metrics import MetricsCalculator, MetricsSnapshot
from ..mobility.position_controller import DummyPositionController
from .actions import actions_to_displacements
from .frames import (
    FrameGenerator,
    FrameGeneratorFactory,
    LogPolarGeometryFactory,
    ObstaclesLayerFactory,
    ScenarioState,
    SignalLayerFactory,
    SquareGeometryFactory,
    get_neighbor_positions,
    get_user_positions,
)
from .rewards import RewardConfig, RewardManager


@dataclass(frozen=True)
class SDQNEnvironmentConfig:
    """Configuration for one vectorized swarm-learning environment."""

    dt: float = 0.1
    num_drones: int = 1
    num_users: int = 1
    drones_speed: float = 10.0
    drones_height: float = 10.0
    boundary_size: FloatRange = 1_000.0
    num_obstacles: IntRange = 0
    metrics_area_samples: int = 0
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        _validate_real(self.dt, name="dt", minimum=0.0, inclusive=False)
        _validate_real(self.drones_speed, name="drones_speed", minimum=0.0)
        _validate_real(self.drones_height, name="drones_height", minimum=0.0)
        _validate_integer(self.num_drones, name="num_drones", minimum=1)
        _validate_integer(self.num_users, name="num_users", minimum=0)
        _validate_integer(
            self.metrics_area_samples,
            name="metrics_area_samples",
            minimum=0,
        )
        _validate_float_range(
            self.boundary_size,
            name="boundary_size",
            minimum=0.0,
            inclusive=False,
        )
        _validate_int_range(self.num_obstacles, name="num_obstacles", minimum=0)

    @property
    def step_displacement(self) -> float:
        return self.drones_speed * self.dt


def _observation_layers() -> list[ObstaclesLayerFactory | SignalLayerFactory]:
    """Create the layers shared by the supported SDQN observations."""

    obstacles = ObstaclesLayerFactory(label="Obstacles", plot_center=False)
    users = SignalLayerFactory(
        tx_positions_getter=get_neighbor_positions,
        rx_positions_getter=get_user_positions,
        coverage_mode="none",
        plot_tx_points=False,
        plot_rx_points=True,
        plot_center=False,
        label="Users coverage",
    )
    return [obstacles, users]


def cartesian_frame_factory() -> FrameGeneratorFactory:
    """Create the two-channel Cartesian observation layout."""

    return FrameGeneratorFactory(
        geometry_factory=SquareGeometryFactory(side_size=84, radius=2_000.0),
        layer_factories=_observation_layers(),
        label="SDQN Cartesian observation",
    )


def default_frame_factory() -> FrameGeneratorFactory:
    """Create the maintained two-channel log-polar observation layout."""

    return FrameGeneratorFactory(
        geometry_factory=LogPolarGeometryFactory(
            num_radial=84,
            num_angular=84,
            min_radius=10.0,
            max_radius=2_000.0,
        ),
        layer_factories=_observation_layers(),
        label="SDQN observation",
    )


class SDQNEnvironment(Simulator):
    """Gym-style vector environment for all drones in one simulation.

    Actions are applied as discrete XY displacements before exactly one core
    simulation step.  Invalid collision attempts are restored first, ensuring
    that the returned snapshot, observations, and metrics all describe the
    same accepted state.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SDQNEnvironmentConfig | None = None,
        *,
        environment: Environment | None = None,
        frame_factory: FrameGeneratorFactory | None = None,
    ) -> None:
        self.config = config or SDQNEnvironmentConfig()
        self._owns_environment = environment is None
        simulation_environment = environment or Environment(obstacles=[])
        self._reset_random_streams()

        if self._owns_environment:
            reset_random_environment(
                simulation_environment,
                self.config.boundary_size,
                self.config.num_obstacles,
                rng=self._environment_rng,
            )
        else:
            simulation_environment.require_boundary()

        agents = self._create_agents(simulation_environment)
        super().__init__(
            environment=simulation_environment,
            agents=agents,
            dt=self.config.dt,
        )

        self.frame_factory = frame_factory or default_frame_factory()
        self.frame_generators = self._create_frame_generators()
        self.reward_manager = RewardManager(
            env=self.environment, config=self.config.reward
        )
        self.metrics_calculator = MetricsCalculator(
            self.environment,
            area_samples=self.config.metrics_area_samples,
            rng=self._metrics_rng,
        )

        self.metrics: MetricsSnapshot | None = None
        self.last_frames: NDArray[np.uint8] | None = None
        self.last_actions: NDArray[np.int32] | None = None
        self.last_collisions = np.zeros(self.num_drones, dtype=np.bool_)

    @property
    def num_envs(self) -> int:
        """Number of per-drone observations/actions in each vectorized step."""

        return self.num_drones

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return self.frame_factory.shape

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        """Reset the scenario and return initial observations and metadata."""

        if seed is not None:
            self._reset_random_streams(seed)
            self.metrics_calculator.rng = self._metrics_rng
            for user in self.users:
                user.rng = self._motion_rng  # type: ignore[attr-defined]
                user.dynamics.rng = self._motion_rng  # type: ignore[attr-defined]

        reset_options = dict(options or {})
        allowed_options = {
            "states",
            "drone_states",
            "user_states",
            "randomize_environment",
            "boundary_size",
            "num_obstacles",
        }
        unknown = reset_options.keys() - allowed_options
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown reset options: {names}.")
        environment_options = {
            "randomize_environment",
            "boundary_size",
            "num_obstacles",
        }
        unsupported = reset_options.keys() & environment_options
        if not self._owns_environment and unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                f"Injected environments do not accept reset options: {names}."
            )

        if self._owns_environment and reset_options.get("randomize_environment", True):
            reset_random_environment(
                self.environment,
                reset_options.get("boundary_size", self.config.boundary_size),
                reset_options.get("num_obstacles", self.config.num_obstacles),
                rng=self._environment_rng,
            )

        states_option = reset_options.get("states")
        if states_option is not None and (
            "drone_states" in reset_options or "user_states" in reset_options
        ):
            raise ValueError(
                "Use either the 'states' mapping or per-agent state options, not both."
            )

        if states_option is None:
            states = self._initial_states(
                drone_states=reset_options.get("drone_states"),
                user_states=reset_options.get("user_states"),
            )
        else:
            states = states_option

        snapshot = super().reset(states)
        self.frame_generators = self._create_frame_generators()
        self.last_actions = None
        self.last_collisions = np.zeros(self.num_drones, dtype=np.bool_)
        self._refresh_outputs()
        assert self.last_frames is not None
        return self.last_frames, self._info(snapshot=snapshot, seed=seed)

    def step(
        self, actions: NDArray[np.integer] | list[int] | tuple[int, ...]
    ) -> tuple[
        NDArray[np.uint8],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict[str, Any],
    ]:
        """Apply one action per drone and advance the core exactly once."""

        action_values = np.asarray(actions)
        expected_shape = (self.num_drones,)
        if action_values.shape != expected_shape:
            raise ValueError(f"Actions must have shape {expected_shape}.")
        directions = actions_to_displacements(action_values)
        self.last_actions = action_values.astype(np.int32, copy=True)

        previous_positions = np.array(
            [drone.state[:3] for drone in self.drones], dtype=np.float64
        )
        proposed_positions = previous_positions.copy()
        proposed_positions[:, :2] += directions * self.config.step_displacement
        proposed_positions[:, 2] = (
            self.environment.get_elevation(proposed_positions[:, :2])
            + self.config.drones_height
        )

        collisions, accepted_positions = self._resolve_collisions(
            previous_positions, proposed_positions
        )
        for drone, position in zip(self.drones, accepted_positions, strict=True):
            drone.state[:3] = position
            drone.state[3:6] = 0.0

        snapshot = super().step()
        self._refresh_outputs()
        assert self.metrics is not None
        assert self.last_frames is not None

        rewards, state_collisions = self.reward_manager.compute_rewards(
            self.drone_states[:, :3], self.user_states[:, :3]
        )
        collisions |= state_collisions
        rewards[collisions] = self.config.reward.collision_penalty

        terminated = collisions.astype(np.bool_, copy=True)
        truncated = np.zeros(self.num_drones, dtype=np.bool_)
        self.last_collisions = terminated.copy()
        return (
            self.last_frames,
            rewards.astype(np.float32, copy=False),
            terminated,
            truncated,
            self._info(snapshot=snapshot),
        )

    def _create_agents(self, environment: Environment) -> AgentsManager:
        agents = AgentsManager()
        for index in range(self.config.num_drones):
            agents.register_agent(
                Drone(
                    agent_id=index,
                    env=environment,
                    controller=DummyPositionController(),
                    provider=DummyNeighborProvider(),
                    swarm_link=None,
                )
            )
        for index in range(self.config.num_users):
            agents.register_agent(
                User(
                    agent_id=self.config.num_drones + index,
                    env=environment,
                    swarm_link=None,
                    rng=self._motion_rng,
                )
            )
        return agents

    def _initial_states(
        self,
        *,
        drone_states: NDArray[np.floating] | None,
        user_states: NDArray[np.floating] | None,
    ) -> dict[str, NDArray[np.float64]]:
        if drone_states is None:
            drone_values = np.zeros((self.num_drones, 6), dtype=np.float64)
            drone_values[:, :3] = sample_positions(
                self.num_drones,
                self.environment,
                altitude=self.config.drones_height,
                rng=self._placement_rng,
            )
        else:
            drone_values = np.asarray(drone_states, dtype=np.float64)

        if user_states is None:
            user_values = np.zeros((self.num_users, 6), dtype=np.float64)
            user_values[:, :3] = sample_positions(
                self.num_users, self.environment, rng=self._placement_rng
            )
        else:
            user_values = np.asarray(user_states, dtype=np.float64)

        return {
            "gcs": np.zeros((0, 6), dtype=np.float64),
            "drone": drone_values,
            "user": user_values,
        }

    def _create_frame_generators(self) -> list[FrameGenerator]:
        return [
            self.frame_factory.create(env=self.environment)
            for _ in range(self.num_drones)
        ]

    def _reset_random_streams(self, seed: int | None = None) -> None:
        environment_seed, placement_seed, motion_seed, metrics_seed = (
            np.random.SeedSequence(seed).spawn(4)
        )
        self._environment_rng = np.random.default_rng(environment_seed)
        self._placement_rng = np.random.default_rng(placement_seed)
        self._motion_rng = np.random.default_rng(motion_seed)
        self._metrics_rng = np.random.default_rng(metrics_seed)

    def _generate_frames(self) -> NDArray[np.uint8]:
        frames = np.empty(
            (self.num_drones, *self.frame_shape),
            dtype=np.uint8,
        )
        drone_positions = self.drone_states[:, :3]
        user_positions = self.user_states[:, :3]
        for index, generator in enumerate(self.frame_generators):
            state = ScenarioState(
                agent_position=drone_positions[index],
                neighbor_positions=np.delete(drone_positions, index, axis=0),
                user_positions=user_positions,
            )
            frames[index] = generator.generate(state, dtype="uint8")
        return frames

    def _refresh_outputs(self) -> None:
        self.metrics = self.metrics_calculator.calculate(
            self.drone_states, self.user_states
        )
        self.last_frames = self._generate_frames()

    def _resolve_collisions(
        self,
        previous: NDArray[np.float64],
        proposed: NDArray[np.float64],
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Reject environment and inter-drone collisions until state is stable."""

        safety_distance = self.config.reward.collision_dist
        accepted = proposed.copy()
        collided = np.zeros(self.num_drones, dtype=np.bool_)

        for _ in range(self.num_drones + 1):
            current = self.reward_manager.min_separation(
                accepted, check_drones_separation=True
            )
            new_collisions = current <= safety_distance
            new_collisions &= ~collided
            if not new_collisions.any():
                break
            collided |= new_collisions
            accepted[new_collisions] = previous[new_collisions]

        return collided, accepted

    def _info(self, *, snapshot: Any, seed: int | None = None) -> dict[str, Any]:
        return {
            "snapshot": snapshot,
            "metrics": self.metrics,
            "collisions": self.last_collisions.copy(),
            "sim_time": self.time,
            "sim_step": self.step_count,
            **({"seed": seed} if seed is not None else {}),
        }


def _validate_real(
    value: Real,
    *,
    name: str,
    minimum: float,
    inclusive: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    invalid = number < minimum if inclusive else number <= minimum
    if invalid:
        qualifier = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be {qualifier} {minimum}.")


def _validate_integer(
    value: Integral,
    *,
    name: str,
    minimum: int,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")


def _validate_float_range(
    value: FloatRange,
    *,
    name: str,
    minimum: float,
    inclusive: bool,
) -> None:
    values = value if isinstance(value, tuple) else (value,)
    if len(values) not in (1, 2):
        raise ValueError(f"{name} must be a value or a two-value range.")
    for item in values:
        _validate_real(
            item,
            name=name,
            minimum=minimum,
            inclusive=inclusive,
        )
    if len(values) == 2 and values[0] > values[1]:
        raise ValueError(f"{name} minimum cannot exceed its maximum.")


def _validate_int_range(
    value: IntRange,
    *,
    name: str,
    minimum: int,
) -> None:
    values = value if isinstance(value, tuple) else (value,)
    if len(values) not in (1, 2):
        raise ValueError(f"{name} must be a value or a two-value range.")
    for item in values:
        _validate_integer(item, name=name, minimum=minimum)
    if len(values) == 2 and values[0] > values[1]:
        raise ValueError(f"{name} minimum cannot exceed its maximum.")
