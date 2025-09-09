from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.image import AxesImage

from sim.math.path_loss_model import rssi_to_signal_quality, signal_strength_map
from sim.simulators import MultiAgentSimulator


class AgentsPlot:
    """Handles plotting and updating GCS, drones, and users."""

    def __init__(self, ax: Axes, sim: MultiAgentSimulator):
        self.ax = ax
        self.sim = sim
        (self.user_points,) = self.ax.plot([], [], "mo", label="users")
        (self.drone_points,) = self.ax.plot([], [], "bx", label="drones")
        (self.gcs_points,) = self.ax.plot([], [], "k*", label="GCS")

    def update(self):
        if self.sim.gcs_states.shape[0] > 0:
            self.gcs_points.set_data(
                self.sim.gcs_states[:, 0], self.sim.gcs_states[:, 1]
            )
        if self.sim.user_states.shape[0] > 0:
            self.user_points.set_data(
                self.sim.user_states[:, 0], self.sim.user_states[:, 1]
            )
        if self.sim.drone_states.shape[0] > 0:
            self.drone_points.set_data(
                self.sim.drone_states[:, 0], self.sim.drone_states[:, 1]
            )
        # TODO: Try without checking states shape


class ObstaclesPlot:
    """Handles boundary and obstacle plotting."""

    def __init__(self, ax: Axes, sim: MultiAgentSimulator):
        self.ax = ax
        self.sim = sim

    def plot(self):
        env = self.sim.environment
        if env.boundary is not None:
            self.ax.plot(*env.boundary.shape.exterior.xy, "r-", label="boundary")
        for i, obs in enumerate(env._obstacles):
            self.ax.fill(
                *obs.shape.exterior.xy,
                alpha=0.25,
                facecolor="red",
                edgecolor="red",
                hatch="///",
                label="obstacles" if i == 0 else None,
            )


BackgroundType = Literal["elevation", "satellite", "fused", "rssi", "none"]


class BackgroundPlot:
    """Handles different background types."""

    def __init__(
        self,
        ax: Axes,
        sim: MultiAgentSimulator,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        background_type: BackgroundType,
        show_colorbar: bool = False,
    ):
        self.ax = ax
        self.sim = sim
        self.background_type = background_type
        self.xlim = xlim
        self.ylim = ylim
        self.background_image: AxesImage | None = None
        self.show_colorbar = show_colorbar

    @property
    def elevation_map(self):
        if self.sim.environment.elevation_map is None:
            raise RuntimeError("Elevation map not configured.")
        return self.sim.environment.elevation_map

    def plot(self):
        if self.background_type == "none":
            return

        elif self.background_type == "rssi":
            self._plot_rssi()

        elif self.background_type == "elevation":
            self._plot_image(self.elevation_map.elevation_img, cmap="terrain")

        elif self.background_type == "satellite":
            self._plot_image(self.elevation_map.satellite_img, cmap=None)

        elif self.background_type == "fused":
            self._plot_fused()

        else:
            raise ValueError(f"Invalid background type: {self.background_type}")

    def _plot_image(self, img: np.ndarray, cmap=None):
        extent = (self.xlim[0], self.xlim[1], self.ylim[1], self.ylim[0])
        self.background_image = self.ax.imshow(
            img,
            extent=extent,
            origin="lower",
            alpha=0.7,
            cmap=cmap,
        )

        if cmap and self.show_colorbar:
            plt.colorbar(self.background_image, ax=self.ax, label="Elevation (m)")

    def _plot_fused(self):
        if self.sim.environment.elevation_map is None:
            raise RuntimeError("Elvation map not configured.")

        self._plot_image(self.sim.environment.elevation_map.fused_img)
        elev = self.sim.environment.elevation_map.elevation_data
        sm = plt.cm.ScalarMappable(
            norm=Normalize(vmin=np.nanmin(elev), vmax=np.nanmax(elev)),
            cmap="terrain",
        )
        sm.set_array([])

        if self.show_colorbar:
            plt.colorbar(sm, ax=self.ax, label="Elevation (m)")

    def _plot_rssi(self):
        xs = np.linspace(self.xlim[0], self.xlim[1], 100)
        ys = np.linspace(self.ylim[0], self.ylim[1], 100)
        heatmap = signal_strength_map(
            self.sim.drone_states[:, :3], xs, ys, f=2412, n=2.4, mode="max"
        )
        heatmap = rssi_to_signal_quality(heatmap, vmin=-80) * 100.0

        if self.background_image is not None:
            self.background_image.set_data(heatmap)
            return

        cmap = plt.cm.get_cmap("turbo", 11)
        cmap.set_under("black")
        norm = BoundaryNorm(boundaries=np.linspace(1e-6, 100, 11), ncolors=10)
        self.background_image = self.ax.imshow(
            heatmap,
            extent=(*self.xlim, *self.ylim),
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=0.7,
        )

        if self.show_colorbar:
            plt.colorbar(self.background_image, ax=self.ax, label="Signal Quality (%)")
