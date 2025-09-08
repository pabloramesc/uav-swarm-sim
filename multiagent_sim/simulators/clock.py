import time
from typing import Optional


class SimulationClock:
    """Handles simulation time, step counting, and synchronization."""

    def __init__(self, dt: float, sync_tolerance: float = 0.1) -> None:
        """Initialize the simulation clock.

        Args:
            dt: Default simulation time step in seconds.
            sync_tolerance: Maximum allowed time difference for synchronization.
        """
        self.dt = dt
        self.sync_tolerance = sync_tolerance
        self.init_time: float | None = None
        self.sim_time = 0.0
        self.sim_step = int(0)

    @property
    def real_time(self) -> float:
        """Real elapsed time since the simulation started"""
        return time.time() - self.init_time if self.init_time else 0.0

    def start(self) -> None:
        """Reset the simulation clock to initial values."""
        self.init_time = time.time()
        self.sim_time = 0.0
        self.sim_step = int(0)

    def tick(self, dt: Optional[float] = None) -> float:
        """Advances the simulation by a time step.

        Args:
            dt: Time step to advance. If None, use default dt.

        Returns:
            The time step used for this tick.
        """
        if self.init_time is None:
            raise RuntimeError("Clock not initiated.")
        step_dt = float(dt) if dt is not None else self.dt
        self.sim_time += step_dt
        self.sim_step += 1
        return step_dt

    def sync(self, t: Optional[float] = None) -> None:
        """Synchronize simulation with external or real time reference.

        Args:
            t: External time reference. If None, use real time.
        """
        target_time = float(t) if t is not None else self.real_time
        delta = self.sim_time - target_time
        if delta > self.sync_tolerance:
            time.sleep(delta)
