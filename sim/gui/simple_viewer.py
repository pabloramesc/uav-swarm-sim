import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..core import Simulator
from .fps_control import FPSController
from .plotters import AgentsPlot, BackgroundPlot, BackgroundType, ObstaclesPlot


class SimpleViewer:
    background_type: BackgroundType  # annotation for Pylance type checker

    def __init__(
        self,
        sim: Simulator,
        limits: tuple[float, float, float, float] | None = None,
        figsize: tuple[float, float] | None = None,
        min_fps: float = 10.0,
        max_fps: float = 60.0,
        background_type: BackgroundType = "rssi",
        show_legend: bool = False,
    ) -> None:
        self.sim = sim
        self.limits = self._calculate_axis_limits(limits)
        self.background_type = background_type
        self.show_legend = show_legend

        plt.ion()
        self.fig = plt.figure(figsize=figsize)
        self.ax: Axes = self.fig.add_subplot(111)
        self._configure_axes()
        self._create_plotters()

        self.fps_control = FPSController(min_fps=min_fps, max_fps=max_fps)
        SimpleViewer.reset(self)

    @property
    def fps(self) -> float:
        return self.fps_control.smooth_fps

    def _create_plotters(self) -> None:
        self.background = BackgroundPlot(
            ax=self.ax,
            sim=self.sim,
            xlim=self.limits[0:2],
            ylim=self.limits[2:4],
            background_type=self.background_type,
            show_colorbar=True,
        )
        self.obstacles = ObstaclesPlot(ax=self.ax, sim=self.sim)
        self.agents = AgentsPlot(ax=self.ax, sim=self.sim)

    def reset(self) -> None:
        self.fps_control.reset()
        self.background.plot()
        self.obstacles.plot()
        self.agents.update()
        if self.show_legend:
            self.ax.legend(loc="upper right")

    def render(self, force: bool = False) -> None:
        if not self.fps_control.need_render(self.sim.time) and not force:
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

    def close(self) -> None:
        plt.close(self.fig)

    def _calculate_axis_limits(
        self,
        limits: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float, float, float]:

        if limits is not None:
            return limits

        env = self.sim.environment

        if env.boundary is not None:
            bounds = env.boundary.bounds

            # Increase limits by 10% to appreciate boundary
            half_width = bounds.width * 0.55  # +10% = 110% -> 110% / 2 = 55%
            half_height = bounds.height * 0.55

            x_center, y_center = bounds.center

            new_lims = (
                x_center - half_width,
                x_center + half_width,
                y_center - half_height,
                y_center + half_height,
            )

            return new_lims

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

        self.fig.tight_layout()
