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
        self.frame_images: list[AxesImage] = []
        super().__init__(sim, xlim, ylim, fig_size, min_fps, max_fps, background_type)
        self.sim: SDQNSimulator = sim

    def _create_axes(self) -> list[Axes]:
        num_channels = self._get_frame_channels()
        total_axes = 1 + num_channels # Main sim plot + frame channels
        self.fig.set_size_inches(4 * total_axes, 4)
        axes = [
            self.fig.add_subplot(1, total_axes, i + 1) for i in range(total_axes)
        ]
        self.ax = axes[0]
        self.frame_axes = axes[1:]
        return axes

    def _init_plots(self) -> None:
        # self._init_frame_images()
        (self.drone0_point,) = self.ax.plot([], [], "rx", label="drone0")
        super()._init_plots()
        if self.ax.get_legend() is not None:
            self.ax.get_legend().remove()
        self.fig.tight_layout()

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
        # Remove old images
        for im in self.frame_images:
            im.remove()
        self.frame_images = []

        frames = self._get_drone_frames()
        labels = self._get_frame_labels()
        cmaps = ["gray", "viridis", "plasma", "magma", "cividis"]

        for i, ax in enumerate(self.frame_axes):
            im = self._init_frame(
                frame=frames[..., 0],
                ax=ax,
                cmap=cmaps[i % len(cmaps)],
                label=labels[i] if i < len(labels) else "Channel {i}",
            )
            self.frame_images.append(im)

    def _update_frame_images(self) -> None:
        if not self.frame_images:
            self._init_frame_images()
            return

        frames = self._get_drone_frames()
        radius = self._get_frame_radius()
        for i, im in enumerate(self.frame_images):
            im.set_data(frames[..., i] / 255.0)
            im.set_extent([-radius, +radius, -radius, +radius])
            im.axes.set_xlim([-radius, +radius])
            im.axes.set_ylim([-radius, +radius])
            
    def _get_drone_frames(self, drone_idx: int = 0) -> np.ndarray:
        return self.sim.sdqn_brain.frames[drone_idx]
    
    def _get_frame_channels(self, iface_idx: int = 0) -> int:
        iface = self.sim.sdqn_brain.ifaces[iface_idx]
        return iface.frame_generator.channels
    
    def _get_frame_labels(self, iface_idx: int = 0) -> list[str]:
        iface = self.sim.sdqn_brain.ifaces[iface_idx]
        return [layer.label for layer in iface.frame_generator.layers]

    def _get_frame_radius(self, iface_idx: int = 0) -> float:
        iface = self.sim.sdqn_brain.ifaces[iface_idx]
        return iface.frame_generator.geometry.radius

    def _init_frame(
        self, frame: np.ndarray, ax: Axes, cmap: str, label: str
    ) -> AxesImage:
        im = ax.imshow(frame / 255.0, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
        self.fig.colorbar(im, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True)
        return im
