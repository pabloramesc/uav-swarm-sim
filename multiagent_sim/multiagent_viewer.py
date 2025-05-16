"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time
from typing import Literal

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from numpy.typing import ArrayLike

from .math.path_loss_model import rssi_to_signal_quality, signal_strength_map
from .multiagent_simulator import MultiAgentSimulator
from .utils.logger import create_logger

AspectRatios = Literal["auto", "equal"]
BackgroundType = Literal["elevation", "rssi", "none"]

matplotlib.use("Qt5Agg")


class MultiAgentViewer:

    def __init__(
        self,
        sim: MultiAgentSimulator,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        fig_size: tuple[float, float] = None,
        min_fps: float = 10.0,
        max_fps: float = 60.0,
        aspect_ratio: AspectRatios = "equal",
        background_type: BackgroundType = "rssi",
    ):
        plt.ion()

        self.sim = sim

        self.xlim = xlim
        self.ylim = ylim
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.aspect_ratio = aspect_ratio
        self.background_type = background_type

        self.background_image: AxesImage = None

        self.fig = plt.figure(figsize=fig_size)
        self.ax = self.fig.add_subplot()

        self.logger = create_logger(name="MultiAgentViewer", level="INFO")

        self.reset()

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

    def reset(self) -> None:
        """
        Resets the viewer to its initial state.

        This method clears the axes, initializes the plot elements, and resets
        the rendering timers.
        """
        self.ax.clear()
        self._reset_timers()
        self._calculate_axis_limits()

        if self.background_type == "elevation":
            self._plot_elevation_map()

        self._initiate_plots()
        self._update_agent_points()
        self._plot_rssi_heatmap()
        self._configure_axes()

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
        self._update_links_lines()

        if self.background_type == "rssi":
            self._plot_rssi_heatmap()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # plt.pause(0.1)

        if verbose:
            self._print_sim_status()

        self.last_render_time = self.time

    def _initiate_plots(self) -> None:
        (self.link_lines,) = self.ax.plot([], [], "g-", lw=0.5, label="springs")
        (self.drone_points,) = self.ax.plot([], [], "bx", label="drones")
        (self.edge_drone_points,) = self.ax.plot([], [], "rx", label="edge drones")
        (self.user_points,) = self.ax.plot([], [], "mo", label="users")
        (self.gcs_points,) = self.ax.plot([], [], "k*", label="GCS")
        self._plot_avoid_regions()

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

    def _update_agent_points(self) -> None:
        self.gcs_states = np.atleast_2d(self.sim.gcs.state)
        self.gcs_points.set_data(self.gcs_states[:, 0], self.gcs_states[:, 1])

        self.user_states = self.sim.users.get_states_array()
        self.user_points.set_data(self.user_states[:, 0], self.user_states[:, 1])

        self.drone_states = self.sim.drones.get_states_array()
        self.drone_points.set_data(
            self.drone_states[~self.sim.edge_drones_mask, 0],
            self.drone_states[~self.sim.edge_drones_mask, 1],
        )
        self.edge_drone_points.set_data(
            self.drone_states[self.sim.edge_drones_mask, 0],
            self.drone_states[self.sim.edge_drones_mask, 1],
        )

    def _update_links_lines(self) -> None:
        links_x, links_y = self._get_links_coords()
        self.link_lines.set_data(links_x, links_y)

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
        self.ax.set_title("Multi-agent EVSM simulation")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_aspect(self.aspect_ratio)
        self.ax.grid(True)
        self.ax.legend(loc="upper right")

        if self.xlim:
            self.ax.set_xlim(*self.xlim)

        if self.ylim:
            self.ax.set_ylim(*self.ylim)

        self.fig.tight_layout()

    def _plot_elevation_map(self) -> None:
        if self.sim.environment.elevation_map is None:
            return

        self.background_image = self.ax.imshow(
            self.sim.environment.elevation_map.elevation_data,
            extent=(
                self.xlim[0],
                self.xlim[1],
                self.ylim[1],
                self.ylim[0],
            ),  # Flip Y-axis
            origin="lower",
            cmap="terrain",
            alpha=0.7,
        )

    def _plot_rssi_heatmap(self) -> None:
        # Generate the grid for the heatmap
        xs = np.linspace(self.xlim[0], self.xlim[1], 100)
        ys = np.linspace(self.ylim[0], self.ylim[1], 100)

        # Calculate the heatmap using the simulator's tx_power_heatmap method
        heatmap = signal_strength_map(
            self.drone_states[:, 0:3], xs, ys, f=2412, n=2.4, mode="max"
        )
        heatmap = rssi_to_signal_quality(heatmap, vmin=-80) * 100.0  # %

        # Plot the heatmap
        if self.background_image is None:
            cmap = plt.cm.get_cmap("turbo", 11)  # 11 discrete colors
            norm = mcolors.BoundaryNorm(boundaries=np.linspace(0, 100, 11), ncolors=10)
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

    def _get_links_coords(self) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        links_x, links_y = [], []
        for drone1_idx in range(self.sim.num_drones):

            drone1_pos = self.drone_states[drone1_idx, 0:3]
            for drone2_idx in range(self.sim.num_drones):
                if not self.sim.links_matrix[drone1_idx, drone2_idx]:
                    continue

                drone2_pos = self.drone_states[drone2_idx, 0:3]
                links_x.extend([drone1_pos[0], drone2_pos[0], None])
                links_y.extend([drone1_pos[1], drone2_pos[1], None])

            if not np.any(self.sim.links_matrix[drone1_idx]):
                self.logger.info(f"Drone {drone1_idx} has no links.")

        return links_x, links_y

    def _need_render(self) -> bool:
        if self.current_fps > self.max_fps:
            return False
        return self.sim.sim_time > self.time or self.current_fps < self.min_fps

    def _reset_timers(self) -> None:
        self.t0 = time.time()
        self.last_render_time = 0.0

    def _print_sim_status(self) -> None:
        ns3_time = self.sim.network_simulator.ns3_time
        ns3_rtt = self.sim.network_simulator.bridge.mean_rtt * 1e3
        print(
            f"real time: {self.time:.2f} s, sim time: {self.sim.sim_time:.2f} s, "
            f"NS-3 time: {ns3_time:.2f} s, NS-3 RTT: {ns3_rtt:.3} ms, FPS: {self.current_fps:.2f}"
        )
        cr = self.sim.area_coverage_ratio()
        print(f"Area coverage ratio: {cr * 100:.2f} %")
