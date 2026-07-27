"""Cadence and lifecycle adapter for the optional ns-3 backend."""

from __future__ import annotations

import math
import time
from numbers import Integral

from numpy.typing import NDArray

from .network_simulator import NetworkSimulator


class NetworkManager:
    """Adapt :class:`NetworkSimulator` to the generic core backend contract."""

    def __init__(
        self,
        num_gcs: int,
        num_drones: int,
        num_users: int,
        *,
        position_update_period: float = 0.1,
        verification_period: float = 1.0,
    ) -> None:
        if not math.isfinite(position_update_period) or position_update_period <= 0:
            raise ValueError("position_update_period must be positive and finite.")
        if not math.isfinite(verification_period) or verification_period <= 0:
            raise ValueError("verification_period must be positive and finite.")

        self.simulator = NetworkSimulator(
            num_gcs=num_gcs,
            num_drones=num_drones,
            num_users=num_users,
        )
        self.position_update_period = float(position_update_period)
        self.verification_period = float(verification_period)
        self._last_position_update: float | None = None
        self._last_verification: float | None = None
        self._started = False
        self._closed = False

    @property
    def ns3_time(self) -> float:
        return self.simulator.ns3_time

    def initialize(self, positions: dict[int, NDArray]) -> None:
        if self._closed:
            raise RuntimeError("Cannot initialize a closed network backend.")
        self._validate_positions(positions)
        self._last_position_update = None
        self._last_verification = None
        self.simulator.reset()
        self.simulator.launch_simulator(max_attempts=2)
        self._started = True
        self.simulator.set_node_positions(positions)
        self.simulator.verify_node_positions()

    def update(self, time: float, positions: dict[int, NDArray]) -> None:
        self._ensure_started()
        self._validate_positions(positions)
        updated_positions = None
        if self._is_due(time, self._last_position_update, self.position_update_period):
            updated_positions = positions
            self._last_position_update = time

        verify = self._is_due(time, self._last_verification, self.verification_period)
        if verify:
            self._last_verification = time

        self.simulator.update(updated_positions, check=verify)

    def wait_until(self, target_time: float, timeout: float) -> None:
        self._ensure_started()
        if (
            not math.isfinite(target_time)
            or target_time < 0.0
            or not math.isfinite(timeout)
            or timeout < 0.0
        ):
            raise ValueError("target_time and timeout must be finite and non-negative.")

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"ns-3 did not reach simulation time {target_time:.6f} "
                    f"within {timeout:.3f} seconds."
                )
            current = self.simulator.bridge.request_ns3_time(timeout=remaining)
            if current + 1e-12 >= target_time:
                return
            time.sleep(min(0.001, remaining))

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._started:
                self.simulator.shutdown_simulator(timeout=1.0)
        finally:
            self._started = False
            self.simulator.bridge.sock.close()
            self._closed = True

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("Network backend is closed.")
        if not self._started:
            raise RuntimeError("Network backend has not been initialized.")

    def _validate_positions(self, positions: dict[int, NDArray]) -> None:
        if any(
            isinstance(agent_id, bool) or not isinstance(agent_id, Integral)
            for agent_id in positions
        ):
            raise ValueError("Network position keys must be integer agent IDs.")
        expected = set(range(self.simulator.num_nodes))
        received = set(positions)
        if received != expected:
            raise ValueError(
                "Network positions must use contiguous agent IDs "
                f"0..{self.simulator.num_nodes - 1}; got {sorted(received)}."
            )

    @staticmethod
    def _is_due(time: float, previous: float | None, period: float) -> bool:
        return previous is None or time + 1e-12 >= previous + period
