"""Matplotlib viewer with the EVSM virtual-spring mesh overlay."""

import numpy as np

from ..evsm.simulator import EVSMSimulator
from .simple_viewer import BackgroundType, SimpleViewer


class EVSMViewer(SimpleViewer):
    def __init__(
        self,
        simulator: EVSMSimulator,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        fig_size: tuple[float, float] | None = None,
        min_fps: float = 10.0,
        max_fps: float = 60.0,
        background_type: BackgroundType = "rssi",
        show_legend: bool = False,
    ) -> None:
        super().__init__(
            sim=simulator,
            limits=xlim + ylim if xlim and ylim else None,
            figsize=fig_size,
            min_fps=min_fps,
            max_fps=max_fps,
            background_type=background_type,
            # SimpleViewer configures axes before creating artists.  Defer the
            # legend until the EVSM spring artist also exists.
            show_legend=False,
        )
        self.show_legend = show_legend
        self.simulator = simulator
        (self.spring_lines,) = self.ax.plot(
            [],
            [],
            color="tab:blue",
            linewidth=0.8,
            alpha=0.55,
            label="EVSM springs",
            zorder=1,
        )
        if show_legend:
            self.ax.legend(loc="upper right")
        self._update_springs()

    def reset(self) -> None:
        super().reset()
        self._update_springs()

    def render(self, force: bool = False) -> None:
        self._update_springs()
        return super().render(force=force)

    def _update_springs(self) -> None:
        x_coordinates, y_coordinates = self._spring_coordinates()
        self.spring_lines.set_data(x_coordinates, y_coordinates)

    def _spring_coordinates(self) -> tuple[list[float], list[float]]:
        states = self.simulator.drone_states
        directed = self.simulator.evsm_monitor.springs_matrix
        undirected = np.logical_or(directed, directed.T)
        starts, ends = np.nonzero(np.triu(undirected, k=1))

        x_coordinates: list[float] = []
        y_coordinates: list[float] = []
        for start, end in zip(starts, ends, strict=True):
            x_coordinates.extend(
                [float(states[start, 0]), float(states[end, 0]), np.nan]
            )
            y_coordinates.extend(
                [float(states[start, 1]), float(states[end, 1]), np.nan]
            )
        return x_coordinates, y_coordinates
