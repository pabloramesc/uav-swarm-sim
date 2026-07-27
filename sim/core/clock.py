"""Simulation time keeping."""

from __future__ import annotations

import math
import time
from numbers import Real


class SimulationClock:
    """Track deterministic simulation time and elapsed wall-clock time.

    Simulation time only changes when :meth:`tick` is called.  Wall-clock time
    is measured with a monotonic clock, so system clock adjustments cannot make
    ``real_time`` jump backwards.
    """

    def __init__(self, dt: float = 0.01, sync_tolerance: float = 0.1) -> None:
        self._dt = self._validate_duration(dt, name="dt", allow_zero=False)
        self._sync_tolerance = self._validate_duration(
            sync_tolerance,
            name="sync_tolerance",
            allow_zero=True,
        )
        self._started_at: float | None = None
        self._time = 0.0
        self._step_count = 0

    @staticmethod
    def _validate_duration(
        value: float,
        *,
        name: str,
        allow_zero: bool,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number.")

        duration = float(value)
        if not math.isfinite(duration):
            raise ValueError(f"{name} must be finite.")
        if duration < 0.0 or (duration == 0.0 and not allow_zero):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}.")
        return duration

    @property
    def dt(self) -> float:
        """Default simulation step in seconds."""

        return self._dt

    @property
    def sync_tolerance(self) -> float:
        return self._sync_tolerance

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""

        return self._time

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def real_time(self) -> float:
        """Elapsed monotonic time since the most recent reset."""

        if self._started_at is None:
            return 0.0
        return max(time.monotonic() - self._started_at, 0.0)

    @property
    def is_running(self) -> bool:
        return self._started_at is not None

    def reset(self, *, start: bool = True) -> None:
        """Reset simulation counters and optionally start wall-time tracking."""

        self._started_at = time.monotonic() if start else None
        self._time = 0.0
        self._step_count = 0

    def start(self) -> None:
        """Start wall-time tracking after external initialization completes."""

        if self.is_running:
            raise RuntimeError("Simulation clock is already running.")
        self._started_at = time.monotonic()

    def tick(self, dt: float | None = None) -> float:
        """Advance simulation time by one validated step.

        Returns the step duration that was applied.
        """

        if not self.is_running:
            raise RuntimeError("Simulation clock has not been reset.")

        step_dt = (
            self._dt
            if dt is None
            else self._validate_duration(dt, name="dt", allow_zero=False)
        )
        self._time += step_dt
        self._step_count += 1
        return step_dt

    def sync(self, reference_time: float | None = None) -> None:
        """Wait when simulation time is ahead of a real or external clock."""

        if not self.is_running:
            raise RuntimeError("Simulation clock has not been reset.")

        target = (
            self.real_time
            if reference_time is None
            else self._validate_duration(
                reference_time,
                name="reference_time",
                allow_zero=True,
            )
        )
        lead = self._time - target
        if lead > self._sync_tolerance:
            time.sleep(lead)
