import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..simulators.simulator import MultiAgentSimulator
from .plotters import BackgroundType, BackgroundPlot, AgentsPlot, ObstaclesPlot
from .fps_control import FPSController


class SimpleViewer:

    def __init__(
        self,
        sim: MultiAgentSimulator,
        limits: tuple[float, float, float, float] | None = None,
        figsize: tuple[float, float] | None = None,
        min_fps: float = 10.0,
        max_fps: float = 60.0,
        background_type: BackgroundType = "rssi",
        show_legend: bool = False,
    ) -> None:
        self.sim = sim
        self.limits = self._calculate_axis_limits(limits)
        self.show_legend = show_legend

        plt.ion()
        self.fig = plt.figure(figsize=figsize)
        self.ax: Axes = self.fig.add_subplot(111)
        self._configure_axes()

        self.fps_control = FPSController(min_fps=min_fps, max_fps=max_fps)

        self.background = BackgroundPlot(
            ax=self.ax,
            sim=self.sim,
            xlim=self.limits[0:2],
            ylim=self.limits[2:4],
            background_type=background_type,
            show_colorbar=True,
        )
        self.obstacles = ObstaclesPlot(ax=self.ax, sim=self.sim)
        self.agents = AgentsPlot(ax=self.ax, sim=self.sim)

    @property
    def fps(self) -> float:
        return self.fps_control.smooth_fps
    
    def initialize(self) -> None:
        self.background.plot()
        self.obstacles.plot()
        self.agents.update()

    def update(self, force: bool = False) -> None:
        if not self.fps_control.need_render(self.sim.clock.sim_time) and not force:
            return
        
        self.agents.update()
        self.background.plot()
        self.fps_control.record_render()
        
          # Redraw without blocking
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def capture_frame(self) -> np.ndarray:
        self.fig.canvas.draw()
        width, height = self.fig.canvas.get_width_height()
        buf = self.fig.canvas.buffer_rgba()  # type: ignore
        img = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
        return img[..., :3].copy()  # Remove alpha channel

    def _calculate_axis_limits(
        self,
        limits: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float, float, float]:

        if limits is not None:
            return limits

        env = self.sim.environment

        if env.boundary is not None:
            return (*env.boundary.bounds.xlim, *env.boundary.bounds.ylim)

        if env.elevation_map is not None:
            bounds = env.elevation_map.bounds
            south_west = env.geo2enu((bounds.bottom, bounds.left, 0.0))
            north_east = env.geo2enu((bounds.top, bounds.right, 0.0))
            return south_west[0], north_east[0], south_west[1], north_east[1]

        raise RuntimeError("Cannot calculate axis limits.")

    def _configure_axes(self) -> None:
        self.ax.set_xlim(*self.limits[0:2])
        self.ax.set_ylim(*self.limits[2:4])

        self.ax.set_title("Multi-agent simulation")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        if self.show_legend:
            self.ax.legend(loc="upper right")

        self.fig.tight_layout()
