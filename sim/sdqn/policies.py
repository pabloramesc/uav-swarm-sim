"""Lightweight policies for SDQN simulation and observation previews."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .actions import NUM_ACTIONS


class SeededRandomPolicy:
    """Select reproducible random actions without loading the ML stack."""

    def __init__(self, *, seed: int | None = None, num_actions: int = NUM_ACTIONS):
        if isinstance(num_actions, bool) or not isinstance(num_actions, Integral):
            raise TypeError("num_actions must be an integer.")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        self.num_actions = int(num_actions)
        self.rng = np.random.default_rng(seed)

    def act(self, frames: np.ndarray) -> NDArray[np.int32]:
        if frames.ndim != 4:
            raise ValueError("frames must have shape (N, H, W, C).")
        return self.rng.integers(
            0,
            self.num_actions,
            size=frames.shape[0],
            dtype=np.int32,
        )
