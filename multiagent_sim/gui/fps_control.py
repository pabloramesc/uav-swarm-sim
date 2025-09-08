import time
from typing import Optional


class FPSController:
    """Controls rendering timing with optional min/max FPS limit."""

    def __init__(
        self, min_fps: Optional[float] = None, max_fps: Optional[float] = None
    ):
        if min_fps is not None and min_fps <= 0:
            raise ValueError("min_fps must be positive")
        
        if max_fps is not None and max_fps <= 0:
            raise ValueError("max_fps must be positive")
        
        if min_fps is not None and max_fps is not None and min_fps > max_fps:
            raise ValueError("min_fps cannot be greater than max_fps values")
        
        self._min_fps = min_fps
        self._max_fps = max_fps
        self.reset()

    def reset(self):
        """Reset the timer and smooth FPS."""
        self._start_time = time.time()
        self._last_render_time = 0.0
        self._smooth_fps = 0.0

    @property
    def real_time(self) -> float:
        """Time since controller was reset."""
        return time.time() - self._start_time

    @property
    def current_fps(self) -> float:
        """Instantaneous FPS based on time since last render."""
        elapsed = self.real_time - self._last_render_time
        return 1.0 / elapsed if elapsed > 0 else 0.0

    @property
    def smooth_fps(self) -> float:
        """Low-pass filtered FPS value."""
        return self._smooth_fps

    def need_render(self, sim_time: float) -> bool:
        """Decide whether a new frame should be rendered.

        Args:
            sim_time: Current simulation time in seconds.

        Returns:
            True if simulation time is not lagging behind real time,
            or if current FPS < min_fps (if set),
            except current FPS > max_fps (if set).
        """
        # Check upper limit
        if self._max_fps is not None and self.current_fps > self._max_fps:
            return False
        # Check lower limit
        if self._min_fps is not None and self.current_fps < self._min_fps:
            return True
        # Catch up with simulation time
        return sim_time > self.real_time

    def record_render(self):
        """Call after rendering to update smooth FPS and last render timestamp."""
        self._smooth_fps = 0.9 * self._smooth_fps + 0.1 * self.current_fps
        self._last_render_time = self.real_time
