"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..core.multiagent_simulator import MultiAgentSimulator
from ..utils.logger import create_logger


class MultiAgentViewer(ABC):

    def __init__(
        self,
        sim: MultiAgentSimulator,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        fig_size: tuple[float, float] = None,
        min_fps: float = 10.0,
        max_fps: float = 60.0,
    ):
        plt.ion()

        self.sim = sim
        self.xlim = xlim
        self.ylim = ylim
        self.min_fps = min_fps
        self.max_fps = max_fps

        self.fig = plt.figure(figsize=fig_size)
        self.axes = self._create_axes()

        self.logger = create_logger(name="MultiAgentViewer", level="INFO")

        self.reset()

    @abstractmethod
    def _create_axes(self) -> list[Axes]:
        pass

    @abstractmethod
    def _init_plots(self) -> None:
        pass

    @abstractmethod
    def _update_plots(self) -> None:
        pass

    def reset(self) -> None:
        self._reset_timers()
        self._init_plots()
        self._update_plots()
        self._render_figure()

    def update(self, force: bool = False) -> float:
        if not (force or self._need_render()):
            return self.fps
        self._update_plots()
        self._render_figure()
        return self.fps

    @property
    def time(self) -> float:
        return time.time() - self.t0

    @property
    def current_fps(self) -> float:
        elapsed_time = self.time - self.last_render_time
        return 1.0 / elapsed_time if elapsed_time > 0.0 else 0.0

    def _reset_timers(self) -> None:
        self.t0 = time.time()
        self.fps = 0.0
        self.last_render_time = 0.0

    def _need_render(self) -> bool:
        if self.current_fps > self.max_fps:
            return False
        return self.sim.sim_time > self.time or self.current_fps < self.min_fps

    def _render_figure(self) -> None:
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
        self.fps = 0.9 * self.fps + 0.1 * self.current_fps
        self.last_render_time = self.time
