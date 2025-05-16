"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time
from typing import Literal

import matplotlib
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from numpy.typing import ArrayLike

from .math.path_loss_model import rssi_to_signal_quality, signal_strength_map
from .multiagent_sdqn_gym import MultiAgentSDQNGym

AspectRatio = Literal["auto", "equal"]

matplotlib.use("Qt5Agg")


class MultiAgentSDQNViewer:
    """
    A viewer for visualizing the MultiDroneSimulator in a 2D environment.

    This class provides real-time visualization of the drone swarm, including
    drone positions, links between drones, and edge drones. It also supports
    rendering boundaries and obstacles in the environment.
    """

    def __init__(
        self,
        sim: MultiAgentSDQNGym,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        fig_size: tuple[float, float] = None,
        min_fps: float = 10.0,
        aspect_ratio: AspectRatio = "equal",
    ):
        plt.ion()

        self.sim = sim
        self.xlim = xlim
        self.ylim = ylim
        self.min_fps = min_fps
        self.max_fps = 60.0
        self.aspect_ratio = aspect_ratio

        self.im1: AxesImage = None
        self.im2: AxesImage = None
        self.im3: AxesImage = None
        self.im4: AxesImage = None

        self.fig = plt.figure(figsize=fig_size)
        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)

        self.reset()

    def reset(self) -> None:
        """
        Resets the viewer to its initial state.

        This method clears the axes, initializes the plot elements, and resets
        the rendering timers.
        """
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        self._reset_timers()
        self._calculate_axis_limits()
        self._initiate_plots()
        self._configure_axes()

        self._update_agent_points()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # plt.pause(1.0)
        self.last_render_time = self.time

    def update(
        self,
        force_render: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Updates the viewer based on the current simulation state.

        This method decides whether to render the visualization based on the
        elapsed time, simulation time, and rendering constraints.

        Parameters
        ----------
        force_render : bool, optional
            If True, forces rendering regardless of constraints (default is False).
        verbose : bool, optional
            If True, prints real-time and simulation-time statistics (default is False).
        """
        if not (force_render or self._need_render()):
            return

        self._update_agent_points()
        self._plot_rssi_heatmap()

        self._plot_frame_ch1()
        self._plot_frame_ch2()
        self._plot_frame_ch3()

        plt.pause(0.01)

        if verbose:
            print(self.viewer_status_str())

        self._update_timers()

    def viewer_status_str(self) -> str:
        return (
            f"Real time: {self.time:.2f} s, "
            f"Sim time: {self.sim.sim_time:.2f} s, "
            f"FPS: {self.fps:.2f}"
        )

    def _initiate_plots(self) -> None:
        (self.drone0_point,) = self.ax1.plot([], [], "rx", label="drone0")
        (self.drone_points,) = self.ax1.plot([], [], "bx", label="drones")
        (self.user_points,) = self.ax1.plot([], [], "mo", label="users")
        self._plot_avoid_regions()

    def _plot_avoid_regions(self) -> None:
        if self.sim.environment.boundary is not None:
            self.ax1.plot(*self.sim.environment.boundary.shape.exterior.coords.xy, "r-")

        for obs in self.sim.environment.obstacles:
            self.ax1.fill(
                *obs.shape.exterior.coords.xy,
                alpha=0.25,
                facecolor="red",
                edgecolor="red",
                hatch="///",
            )

    def _update_agent_points(self) -> None:
        self.drone_states = self.sim.drones.get_states_array()
        self.drone0_point.set_data(
            self.drone_states[0:1, 0],
            self.drone_states[0:1, 1],
        )
        self.drone_points.set_data(
            self.drone_states[1:, 0],
            self.drone_states[1:, 1],
        )
        self.user_states = self.sim.users.get_states_array()
        self.user_points.set_data(
            self.user_states[:, 0],
            self.user_states[:, 1],
        )

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

    def _configure_axes(self) -> None:
        self.ax1.set_title("Simulation 2D View")
        self.ax1.set_xlabel("X (m)")
        self.ax1.set_ylabel("Y (m)")
        self.ax1.set_aspect(self.aspect_ratio)
        self.ax1.grid(True)
        self.ax1.legend(loc="upper right")

        self.ax2.set_title("Collision Heatmap")
        self.ax2.set_xlabel("X (pixels)")
        self.ax2.set_ylabel("Y (pixels)")

        self.ax3.set_title("Drones Heatmap")
        self.ax3.set_xlabel("X (pixels)")
        self.ax3.set_ylabel("Y (pixels)")

        self.ax4.set_title("Coverage Heatmap")
        self.ax4.set_xlabel("X (pixels)")
        self.ax4.set_ylabel("Y (pixels)")

        self.fig.tight_layout()

        if self.xlim:
            self.ax1.set_xlim(*self.xlim)

        if self.ylim:
            self.ax1.set_ylim(*self.ylim)

    def _plot_rssi_heatmap(self) -> None:
        # Generate the grid for the heatmap
        xs = np.linspace(self.xlim[0], self.xlim[1], 100)
        ys = np.linspace(self.ylim[0], self.ylim[1], 100)

        # Calculate the heatmap using the simulator's tx_power_heatmap method
        rssi = signal_strength_map(
            self.drone_states[:, 0:3], xs, ys, f=2412, n=2.4, mode="max"
        )
        heatmap = rssi_to_signal_quality(rssi, vmin=-80.0) * 100.0

        # Plot the heatmap
        if self.im1 is None:
            cmap = plt.cm.get_cmap("turbo", 11)  # 11 discrete colors
            norm = mcolors.BoundaryNorm(boundaries=np.linspace(0, 100, 11), ncolors=10)
            self.im1 = self.ax1.imshow(
                heatmap,
                extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]],
                origin="lower",
                cmap=cmap,
                norm=norm,
                alpha=0.7,
            )
            plt.colorbar(self.im1, ax=self.ax1, label="Signal Quality (%)")
        else:
            self.im1.set_data(heatmap)

    def _plot_frame_ch1(self, drone_id: int = 0) -> None:
        frame = self.sim.sdqn_agent.last_frames[drone_id, ..., 0] / 255.0
        if self.im2 is None:
            self.im2 = self.ax2.imshow(
                frame, origin="lower", cmap="gray", vmin=0.0, vmax=1.0
            )
            plt.colorbar(self.im2, ax=self.ax2, label="Collision Risk")
        else:
            self.im2.set_data(frame)

    def _plot_frame_ch2(self, drone_id: int = 0) -> None:
        frame = self.sim.sdqn_agent.last_frames[drone_id, ..., 1] / 255.0
        if self.im3 is None:
            self.im3 = self.ax3.imshow(
                frame, origin="lower", cmap="turbo", vmin=0.0, vmax=1.0
            )
            plt.colorbar(self.im3, ax=self.ax3, label="Signal Quality")
        else:
            self.im3.set_data(frame)

    def _plot_frame_ch3(self, drone_id: int = 0) -> None:
        frame = self.sim.sdqn_agent.last_frames[drone_id, ..., 2] / 255.0
        if self.im4 is None:
            self.im4 = self.ax4.imshow(
                frame, origin="lower", cmap="turbo", vmin=0.0, vmax=1.0
            )
            plt.colorbar(self.im4, ax=self.ax4, label="Signal Quality")
        else:
            self.im4.set_data(frame)

    def _need_render(self) -> bool:
        if self.current_fps > self.max_fps:
            return False
        return self.sim.sim_time > self.time or self.current_fps < self.min_fps

    def _reset_timers(self) -> None:
        self.t0 = time.time()
        self.fps = 0.0
        self.last_render_time = 0.0

    def _update_timers(self) -> None:
        self.fps = 0.1 * self.fps + 0.9 * self.current_fps
        self.last_render_time = self.time

    @property
    def time(self) -> float:
        return time.time() - self.t0

    @property
    def time_since_render(self) -> float:
        return self.time - self.last_render_time

    @property
    def current_fps(self) -> float:
        if self.time_since_render > 0.0:
            return 1.0 / self.time_since_render
        return 0.0
