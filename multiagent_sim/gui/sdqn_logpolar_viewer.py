"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh

from ..core.sdqn_simulator import SDQNSimulator
from .simple_viewer import BackgroundType, SimpleViewer
from ..sdqn.frame_generators import FrameGenerator, LogPolarFrameGenerator

"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.scale import LogScale

from ..core.sdqn_simulator import SDQNSimulator
from .simple_viewer import BackgroundType, SimpleViewer


class SDQNGridViewer(SimpleViewer):

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
        for i, im in enumerate(self.frame_images):
            im.set_data(frames[..., i].T / 255.0)

    def _get_drone_frames(self, drone_idx: int = 0) -> np.ndarray:
        return self.sim.sdqn_brain.last_frames[drone_idx]

    def _get_frame_labels(self, iface_idx: int = 0) -> list[str]:
        iface = self.sim.sdqn_brain.ifaces[iface_idx]
        return iface.frame_generator.channel_names

    def _get_frame_generator(self, iface_idx: int = 0) -> FrameGenerator:
        iface = self.sim.sdqn_brain.ifaces[iface_idx]
        return iface.frame_generator

    def _init_frame(
        self, frame: np.ndarray, ax: Axes, cmap: str, label: str
    ) -> AxesImage:
        generator = self._get_frame_generator()
        if not isinstance(generator, LogPolarFrameGenerator):
            raise ValueError("Frame generator is not log-polar")
        log_r_min = np.log10(generator.min_radius)
        log_r_max = np.log10(generator.max_radius)

        im = ax.imshow(
            frame.T / 255.0,
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            extent=[-np.pi, +np.pi, log_r_min, log_r_max],
        )
        plt.colorbar(im, ax=ax)

        ax.set_title(label)
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel("R (km)")

        # Set x-ticks
        ax.set_xticks(np.linspace(-np.pi, np.pi, 9))
        ax.set_xticklabels(
            [f"{int(np.degrees(t))}°" for t in np.linspace(-np.pi, np.pi, 9)]
        )
        ax.invert_xaxis()

        # Set y-ticks
        r_ticks = np.logspace(log_r_min, log_r_max, num=6)
        log_r_ticks = np.log10(r_ticks)
        ax.set_yticks(log_r_ticks)
        ax.set_yticklabels(
            [f"{r:.1f} m" if r < 1000 else f"{r/1000:.1f} km" for r in r_ticks]
        )

        ax.grid(True)
        ax.set_aspect("auto")
        return im


class SDQNLogPolarViewer(SimpleViewer):

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
        self.frame_meshes: list[QuadMesh] = None
        super().__init__(sim, xlim, ylim, fig_size, min_fps, max_fps, background_type)
        self.sim: SDQNSimulator = sim

    def _create_axes(self) -> list[Axes]:
        # self.ax = self.fig.add_subplot(221)
        # self.ax1 = self.fig.add_subplot(222, projection="polar")
        # self.ax2 = self.fig.add_subplot(223, projection="polar")
        # self.ax3 = self.fig.add_subplot(224, projection="polar")
        self.fig.set_size_inches(16, 4)
        self.ax = self.fig.add_subplot(141)
        self.ax1 = self.fig.add_subplot(142, projection="polar")
        self.ax2 = self.fig.add_subplot(143, projection="polar")
        self.ax3 = self.fig.add_subplot(144, projection="polar")
        return [self.ax, self.ax1, self.ax2, self.ax3]

    def _init_plots(self) -> None:
        # self._init_frame_images()
        (self.drone0_point,) = self.ax.plot([], [], "rx", label="drone0")
        super()._init_plots()
        if self.ax.get_legend() is not None:
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
        if self.frame_meshes is not None:
            for pm in self.frame_meshes:
                pm.remove()
            self.frame_meshes = None
        frames = self._get_drone_frames()
        labels = self._get_frame_labels()
        pm1 = self._init_polar_mesh(
            frames[..., 0], ax=self.ax1, cmap="gray", label=labels[0]
        )
        pm2 = self._init_polar_mesh(
            frames[..., 1], ax=self.ax2, cmap="viridis", label=labels[1]
        )
        pm3 = self._init_polar_mesh(
            frames[..., 2], ax=self.ax3, cmap="plasma", label=labels[2]
        )
        self.frame_meshes = [pm1, pm2, pm3]

    def _update_frame_images(self) -> None:
        if self.frame_meshes is None:
            self._init_frame_images()
            return
        frames = self._get_drone_frames()
        for i, pm in enumerate(self.frame_meshes):
            pm.set_array(frames[..., i] / 255.0)

    def _get_drone_frames(self, drone_idx: int = 0) -> np.ndarray:
        return self.sim.sdqn_brain.last_frames[drone_idx]

    def _get_frame_labels(self, iface_idx: int = 0) -> list[str]:
        iface = self.sim.sdqn_brain.ifaces[iface_idx]
        return iface.frame_generator.channel_names

    def _get_frame_generator(self, iface_idx: int = 0) -> FrameGenerator:
        iface = self.sim.sdqn_brain.ifaces[iface_idx]
        return iface.frame_generator

    def _init_polar_mesh(
        self, frame: np.ndarray, ax: Axes, cmap: str, label: str
    ) -> QuadMesh:
        generator = self._get_frame_generator()
        if not isinstance(generator, LogPolarFrameGenerator):
            raise ValueError("Frame generator is not log-polar")
        r_edges, theta_edges = generator.get_logpolar_mesh_edges()
        pm = ax.pcolormesh(theta_edges, r_edges, frame / 255.0, cmap=cmap, vmin=0.0, vmax=1.0)
        plt.colorbar(pm, ax=ax)
        
        ax.set_rscale(LogScale(ax, base=np.e))
        log_r_min = np.log(generator.min_radius)
        log_r_max = np.log(generator.max_radius)
        r_tick_values = np.exp(np.linspace(log_r_min, log_r_max, 6))
        ax.set_rticks(r_tick_values)
        ax.set_rlabel_position(-45)  # Position the radial labels
        
        r_tick_labels = [f"{r:.1f} m" if r < 1000 else f"{r/1000:.1f} km" for r in r_tick_values]
        ax.set_yticklabels(r_tick_labels)
        
        return pm
