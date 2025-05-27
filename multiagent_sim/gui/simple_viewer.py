"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from typing import Literal

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from ..core.multiagent_simulator import MultiAgentSimulator
from ..math.path_loss_model import rssi_to_signal_quality, signal_strength_map
from .multiagent_viewer import MultiAgentViewer

BackgroundType = Literal["elevation", "satellite", "fused", "rssi", "none"]


class SimpleViewer(MultiAgentViewer):

    def __init__(
        self,
        sim: MultiAgentSimulator,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        fig_size: tuple[float, float] = None,
        min_fps: float = 10.0,
        max_fps: float = 60.0,
        background_type: BackgroundType = "rssi",
    ) -> None:
        self.background_type: BackgroundType = background_type
        self.background_image: AxesImage = None
        super().__init__(sim, xlim, ylim, fig_size, min_fps, max_fps)

    def _create_axes(self) -> list[Axes]:
        self.ax = self.fig.add_subplot(111)
        return [self.ax]

    def _init_plots(self) -> None:
        """
        Initiate GCS, drones, and users. Plot avoid regions and elevetion map
        if needed.
        """
        (self.drone_points,) = self.ax.plot([], [], "bx", label="drones")
        (self.user_points,) = self.ax.plot([], [], "mo", label="users")
        (self.gcs_points,) = self.ax.plot([], [], "k*", label="GCS")

        self._plot_avoid_regions()
        self._configure_axes()

        if self.background_type == "none":
            pass
        elif self.background_type == "rssi":
            self._plot_rssi_heatmap()
        elif self.background_type == "elevation":
            self._plot_elevation_img()
        elif self.background_type == "satellite":
            self._plot_satellite_img()
        elif self.background_type == "fused":
            self._plot_fused_map()
        else:
            raise ValueError("Invalid background type option:", self.background_type)


    def _update_plots(self) -> None:
        self._update_agent_points()
        
        if self.background_type == "rssi":
            self._plot_rssi_heatmap()

    def _plot_avoid_regions(self) -> None:
        if self.sim.environment.boundary is not None:
            self.ax.plot(
                *self.sim.environment.boundary.shape.exterior.coords.xy,
                "r-",
                label="boundaries",
            )

        for i, obs in enumerate(self.sim.environment.obstacles):
            self.ax.fill(
                *obs.shape.exterior.coords.xy,
                alpha=0.25,
                facecolor="red",
                edgecolor="red",
                hatch="///",
                label="obstacles" if i == 0 else None,
            )

    def _plot_elevation_img(self) -> None:
        self.background_image = self.ax.imshow(
            self.sim.environment.elevation_map.elevation_img,
            extent=(*self.xlim, *self.ylim[::-1]),  # Flip Y-axis
            origin="lower",
            cmap="terrain",
            alpha=0.7,
        )
        plt.colorbar(self.background_image, ax=self.ax, label="Elevation (m)")

    def _plot_satellite_img(self) -> None:
        self.background_image = self.ax.imshow(
            self.sim.environment.elevation_map.satellite_img,
            extent=(*self.xlim, *self.ylim[::-1]),  # Flip Y-axis
            origin="lower",
            alpha=0.7,
        )
        # no colorbar for sat

    def _plot_fused_map(self) -> None:
        self.background_image = self.ax.imshow(
            self.sim.environment.elevation_map.fused_img,
            extent=(*self.xlim, *self.ylim[::-1]),  # Flip Y-axis
            origin="lower",
            alpha=0.7,
        )
        # no colorbar for sat

    def _plot_rssi_heatmap(self) -> None:
        # Generate the grid for the heatmap
        xs = np.linspace(self.xlim[0], self.xlim[1], 100)
        ys = np.linspace(self.ylim[0], self.ylim[1], 100)

        # Calculate the heatmap using the simulator's tx_power_heatmap method
        heatmap = signal_strength_map(
            self.sim.drone_states[:, 0:3], xs, ys, f=2412, n=2.4, mode="max"
        )
        heatmap = rssi_to_signal_quality(heatmap, vmin=-80) * 100.0  # %

        # Plot the heatmap
        if self.background_image is None:
            cmap = plt.cm.get_cmap("turbo", 11)  # 11 discrete colors
            cmap.set_under("black")  # Color for values below the first boundary
            norm = mcolors.BoundaryNorm(boundaries=np.linspace(1e-6, 100, 11), ncolors=10)
            self.background_image = self.ax.imshow(
                heatmap,
                extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]],
                origin="lower",
                cmap=cmap,
                norm=norm,
                alpha=0.7,
            )
            plt.colorbar(self.background_image, ax=self.ax, label="Signal Quality (%)")
        else:
            self.background_image.set_data(heatmap)

    def _update_agent_points(self) -> None:
        gcs_states = np.atleast_2d(self.sim.gcs.state)
        self.gcs_points.set_data(gcs_states[:, 0], gcs_states[:, 1])
        self.user_points.set_data(
            self.sim.user_states[:, 0], self.sim.user_states[:, 1]
        )
        self.drone_points.set_data(
            self.sim.drone_states[:, 0], self.sim.drone_states[:, 1]
        )

    def _configure_axes(self) -> None:
        self._calculate_axis_limits()
        self.ax.set_title("Multi-agent simulation")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.legend(loc="upper right")

        if self.xlim:
            self.ax.set_xlim(*self.xlim)

        if self.ylim:
            self.ax.set_ylim(*self.ylim)

        self.fig.tight_layout()

    def _calculate_axis_limits(self) -> None:
        if self.sim.environment.elevation_map is None:
            x, y = self.sim.environment.boundary.shape.exterior.xy
            new_xlim = (min(x) - 0.1 * np.ptp(x), max(x) + 0.1 * np.ptp(x))
            new_ylim = (min(y) - 0.1 * np.ptp(y), max(y) + 0.1 * np.ptp(y))
            self.xlim = new_xlim if self.xlim is None else self.xlim
            self.ylim = new_ylim if self.ylim is None else self.ylim
            return

        bounds = self.sim.environment.elevation_map.bounds
        south_west = self.sim.environment.geo2enu((bounds.bottom, bounds.left, 0.0))
        north_east = self.sim.environment.geo2enu((bounds.top, bounds.right, 0.0))
        self.xlim = (south_west[0], north_east[0]) if self.xlim is None else self.xlim
        self.ylim = (south_west[1], north_east[1]) if self.ylim is None else self.ylim
