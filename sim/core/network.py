"""Interface expected from optional network simulation backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NetworkBackend(Protocol):
    """Lifecycle boundary between the simulation core and a network backend."""

    def initialize(self, positions: dict[int, np.ndarray]) -> None:
        """Start the backend and set initial agent positions."""

        ...

    def update(self, time: float, positions: dict[int, np.ndarray]) -> None:
        """Advance backend state using the latest simulation positions."""

        ...

    def wait_until(self, target_time: float, timeout: float) -> None:
        """Wait for the backend to reach ``target_time`` or raise on timeout."""

        ...

    def close(self) -> None:
        """Release backend resources."""

        ...
