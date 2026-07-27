"""Read-only simulation transition results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

_STANDARD_AGENT_TYPES = ("gcs", "drone", "user")


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    """A detached view of simulator state after one transition.

    Arrays are copied and marked read-only.  This prevents normal accidental
    mutation while keeping NumPy access inexpensive.
    """

    time: float
    step_count: int
    states: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if isinstance(self.time, bool) or not isinstance(self.time, Real):
            raise TypeError("time must be a real number.")
        normalized_time = float(self.time)
        if not math.isfinite(normalized_time) or normalized_time < 0.0:
            raise ValueError("time must be finite and non-negative.")

        if isinstance(self.step_count, bool) or not isinstance(
            self.step_count, Integral
        ):
            raise TypeError("step_count must be an integer.")
        normalized_step_count = int(self.step_count)
        if normalized_step_count < 0:
            raise ValueError("step_count must be non-negative.")

        if not isinstance(self.states, Mapping):
            raise TypeError("states must be a mapping keyed by agent type.")

        copied: dict[str, np.ndarray] = {}
        ordered_types = (*_STANDARD_AGENT_TYPES, *self.states.keys())
        for agent_type in dict.fromkeys(ordered_types):
            if not isinstance(agent_type, str):
                raise TypeError("State mapping keys must be strings.")
            source = self.states.get(agent_type)
            if source is None:
                source = np.empty((0, 6), dtype=float)
            array = np.array(source, copy=True)
            array.setflags(write=False)
            copied[agent_type] = array

        object.__setattr__(self, "time", normalized_time)
        object.__setattr__(self, "step_count", normalized_step_count)
        object.__setattr__(self, "states", MappingProxyType(copied))

    @property
    def gcs_states(self) -> np.ndarray:
        return self.states["gcs"]

    @property
    def drone_states(self) -> np.ndarray:
        return self.states["drone"]

    @property
    def user_states(self) -> np.ndarray:
        return self.states["user"]
