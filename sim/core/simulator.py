"""Generic simulation transition coordinator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .clock import SimulationClock
from .network import NetworkBackend
from .snapshot import SimulationSnapshot

if TYPE_CHECKING:
    from ..agents.agent import Agent
    from ..agents.agents_manager import AgentsManager
    from ..agents.agents_registry import AgentsRegistry


class Simulator:
    """Coordinate agents, time and an optional network backend.

    The core intentionally knows nothing about controllers, metrics, rendering
    or learning algorithms.  A call to :meth:`step` is the single owner of a
    simulation transition.
    """

    def __init__(
        self,
        environment: Any,
        agents: AgentsManager,
        dt: float = 0.01,
        network: NetworkBackend | None = None,
    ) -> None:
        self.environment = environment
        self.agents = agents
        self.network = network
        self.clock = SimulationClock(dt)
        self._snapshot: SimulationSnapshot | None = None
        self._closed = False

    @property
    def dt(self) -> float:
        return self.clock.dt

    @property
    def time(self) -> float:
        return self.clock.time

    @property
    def step_count(self) -> int:
        return self.clock.step_count

    @property
    def real_time(self) -> float:
        return self.clock.real_time

    @property
    def snapshot(self) -> SimulationSnapshot:
        if self._snapshot is None:
            raise RuntimeError("Simulator has not been reset.")
        return self._snapshot

    @property
    def states(self) -> Mapping[str, np.ndarray]:
        return self.snapshot.states

    @property
    def gcs_states(self) -> np.ndarray:
        return self.snapshot.gcs_states

    @property
    def drone_states(self) -> np.ndarray:
        return self.snapshot.drone_states

    @property
    def user_states(self) -> np.ndarray:
        return self.snapshot.user_states

    @property
    def all_agents(self) -> AgentsRegistry:
        return self.agents.all_agents

    @property
    def gcs(self) -> AgentsRegistry:
        return self.agents.gcs

    @property
    def drones(self) -> AgentsRegistry:
        return self.agents.drones

    @property
    def users(self) -> AgentsRegistry:
        return self.agents.users

    @property
    def num_agents(self) -> int:
        return self.all_agents.size

    @property
    def num_gcs(self) -> int:
        return self.gcs.size

    @property
    def num_drones(self) -> int:
        return self.drones.size

    @property
    def num_users(self) -> int:
        return self.users.size

    def reset(
        self,
        states: Mapping[str, np.ndarray] | np.ndarray,
    ) -> SimulationSnapshot:
        """Reset time and initialize every registered agent.

        Prefer a mapping keyed by ``"gcs"``, ``"drone"`` and ``"user"``.
        A full array in global registration order remains supported for older
        callers.
        """

        self._ensure_open()
        assignments = self._validated_assignments(states)
        self._snapshot = None

        # Populate all dynamics first.  Drone controllers can then inspect
        # every neighbor during their own initialize hook, regardless of
        # registration order.
        for agent, state in assignments:
            agent.dynamics.state = state.copy()
            agent.time = 0.0

        self.clock.reset(start=False)
        for agent, state in assignments:
            agent.initialize(state=state.copy(), time=self.time)

        if self.network is not None:
            self.network.initialize(self._positions())

        # Backend launch can be expensive and is not part of simulated runtime.
        self.clock.start()
        self._snapshot = self._make_snapshot()
        return self._snapshot

    def step(
        self,
        dt: float | None = None,
        sync: bool = False,
    ) -> SimulationSnapshot:
        """Advance every agent exactly once and return the resulting snapshot."""

        self._ensure_open()
        if self._snapshot is None:
            raise RuntimeError("Simulator must be reset before stepping.")

        step_dt = self.clock.tick(dt)
        self._snapshot = None
        for agent in self.all_agents:
            agent.prepare_step(step_dt)
        for agent in self.all_agents:
            agent.update(step_dt)

        if self.network is not None:
            self.network.update(self.time, self._positions())

        if sync:
            self.sync()

        self._snapshot = self._make_snapshot()
        return self._snapshot

    def sync(self) -> None:
        """Synchronize with wall time and give an optional backend time to catch up."""

        if not self.clock.is_running:
            raise RuntimeError("Simulator must be reset before synchronization.")
        self.clock.sync()
        if self.network is not None:
            self.network.wait_until(
                target_time=self.time,
                timeout=self.clock.sync_tolerance,
            )

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.network is not None:
                self.network.close()
        finally:
            self._closed = True

    def __enter__(self) -> Simulator:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False

    def _validated_assignments(
        self,
        states: Mapping[str, np.ndarray] | np.ndarray,
    ) -> list[tuple[Agent, np.ndarray]]:
        if isinstance(states, Mapping):
            unknown = set(states).difference(self.agents.registries)
            if unknown:
                names = ", ".join(sorted(map(str, unknown)))
                raise ValueError(f"Unknown agent state types: {names}.")

            assignments: list[tuple[Agent, np.ndarray]] = []
            for agent_type, registry in self.agents.registries.items():
                if agent_type not in states:
                    if registry.size:
                        raise ValueError(
                            f"Missing states for {registry.size} "
                            f"'{agent_type}' agent(s)."
                        )
                    continue
                assignments.extend(
                    self._assignments_for_registry(
                        registry,
                        states[agent_type],
                        label=agent_type,
                    )
                )
            return assignments

        array = self._as_state_array(states, label="all")
        if array.shape[0] != self.num_agents:
            raise ValueError(
                "States for all agents must have "
                f"{self.num_agents} rows; got {array.shape[0]}."
            )
        return self._validate_agent_rows(list(self.all_agents), array, label="all")

    def _assignments_for_registry(
        self,
        registry: AgentsRegistry,
        states: np.ndarray,
        *,
        label: str,
    ) -> list[tuple[Agent, np.ndarray]]:
        array = self._as_state_array(states, label=label)
        if array.shape[0] != registry.size:
            raise ValueError(
                f"States for '{label}' agents must have {registry.size} rows; "
                f"got {array.shape[0]}."
            )
        return self._validate_agent_rows(list(registry), array, label=label)

    @staticmethod
    def _as_state_array(states: object, *, label: str) -> np.ndarray:
        try:
            array = np.asarray(states, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"States for '{label}' agents must be numeric.") from exc
        if array.ndim != 2:
            raise ValueError(
                f"States for '{label}' agents must be a two-dimensional array."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"States for '{label}' agents must be finite.")
        return array

    @staticmethod
    def _validate_agent_rows(
        agents: list[Agent],
        states: np.ndarray,
        *,
        label: str,
    ) -> list[tuple[Agent, np.ndarray]]:
        assignments: list[tuple[Agent, np.ndarray]] = []
        for index, (agent, state) in enumerate(zip(agents, states, strict=True)):
            state_copy = np.array(state, dtype=float, copy=True)
            try:
                agent.dynamics.check_state(state_copy)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid state {index} for '{label}' agents: {exc}"
                ) from exc
            assignments.append((agent, state_copy))
        return assignments

    def _make_snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            time=self.time,
            step_count=self.step_count,
            states=self.agents.get_states(),
        )

    def _positions(self) -> dict[int, np.ndarray]:
        return {
            agent_id: np.array(position, copy=True)
            for agent_id, position in self.all_agents.get_positions_dict().items()
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Simulator is closed.")
