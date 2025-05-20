"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from ..core.sdqn_simulator import SDQNSimulator
from .simple_viewer import BackgroundType, SimpleViewer


class SDQNViewer(SimpleViewer):

    def __init__(
        self,
        sim: SDQNSimulator,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        fig_size: tuple[float, float] = None,
        min_fps: float = 10.0,
        max_fps: float = 60.0,
        background_type: BackgroundType = "rssi",
    ):
        self.frame_images: list[AxesImage] = None
        super().__init__(sim, xlim, ylim, fig_size, min_fps, max_fps, background_type)
        self.sim: SDQNSimulator = sim

    def _create_axes(self) -> list[Axes]:
        self.ax = self.fig.add_subplot(221)
        self.ax1 = self.fig.add_subplot(222)
        self.ax2 = self.fig.add_subplot(223)
        self.ax3 = self.fig.add_subplot(224)
        return [self.ax, self.ax1, self.ax2, self.ax3]

    def _init_plots(self) -> None:
        self._init_frame_images()
        (self.drone0_point,) = self.ax.plot([], [], "rx", label="drone0")
        super()._init_plots()
        self.ax.get_legend().remove()

    def _update_plots(self):
        self._update_frame_images()
        super()._update_plots()

    def _update_agent_points(self):
        super()._update_agent_points()
        self.drone_points.set_data(
            self.sim.drone_states[1:, 0], self.sim.drone_states[1:, 1]
        )
        self.drone0_point.set_data(
            self.sim.drone_states[:1, 0], self.sim.drone_states[:1, 1]
        )

    def _init_frame_images(self) -> None:
        frames = self._get_drone_frames()
        labels = self._get_frame_labels()
        im1 = self._init_frame(
            frames[..., 0], ax=self.ax1, cmap="gray", label=labels[0]
        )
        im2 = self._init_frame(
            frames[..., 1], ax=self.ax2, cmap="viridis", label=labels[1]
        )
        im3 = self._init_frame(
            frames[..., 2], ax=self.ax3, cmap="plasma", label=labels[2]
        )
        self.frame_images = [im1, im2, im3]

    def _update_frame_images(self) -> None:
        frames = self._get_drone_frames()
        fr = self._get_frame_radius()
        for i, im in enumerate(self.frame_images):
            im.set_data(frames[..., i] / 255.0)
            im.set_extent([-fr, +fr, -fr, +fr])
            im.axes.set_xlim([-fr, +fr])
            im.axes.set_ylim([-fr, +fr])

    def _get_drone_frames(self, drone_idx: int = 0) -> np.ndarray:
        return self.sim.sdqn_brain.last_frames[drone_idx]

    def _get_frame_labels(self, iface_idx: int = 0) -> list[str]:
        iface = self.sim.sdqn_brain.ifaces[iface_idx]
        return iface.frame_generator.channel_names

    def _get_frame_radius(self, iface_idx: int = 0) -> float:
        iface = self.sim.sdqn_brain.ifaces[iface_idx]
        return iface.frame_generator.frame_radius

    def _init_frame(
        self, frame: np.ndarray, ax: Axes, cmap: str, label: str
    ) -> AxesImage:
        im = ax.imshow(frame / 255.0, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True)
        return im
