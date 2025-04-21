"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike

from simulator.multidrone_gym_dqns import MultidroneGymDQNS

from ..math.path_loss_model import signal_strength, signal_strength_map

AspectRatios = Literal["auto", "equal"]


class MultiDroneViewerDQNS:
    """
    A viewer for visualizing the MultiDroneSimulator in a 2D environment.

    This class provides real-time visualization of the drone swarm, including
    drone positions, links between drones, and edge drones. It also supports
    rendering boundaries and obstacles in the environment.
    """

    def __init__(
        self,
        sim: MultidroneGymDQNS,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        zlim: tuple[float, float] = None,
        fig_size: tuple[float, float] = None,
        min_fps: float = 10.0,
        aspect_ratio: AspectRatios = "equal",
    ):
        self.sim = sim
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.min_fps = min_fps
        self.max_fps = 60.0
        if aspect_ratio not in AspectRatios.__args__:
            raise ValueError("Aspect ratio must be 'auto' or 'equal'")
        self.aspect_ratio = aspect_ratio

        self.t0: float = None
        self.fps: float = None
        self.last_render_time: float = None

        self.drone_points: Line2D = None
        self.signal_heatmap_image: AxesImage = None
        self.drone_frame_image: AxesImage = None

        self.fig = plt.figure(figsize=fig_size)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        self.reset()

    @property
    def time(self) -> float:
        return time.time() - self.t0

    @property
    def elapsed_time(self) -> float:
        return self.time - self.last_render_time

    def reset(self) -> None:
        """
        Resets the viewer to its initial state.

        This method clears the axes, initializes the plot elements, and resets
        the rendering timers.
        """
        self.ax1.clear()
        self.ax2.clear()

        self._reset_timers()
        self._calculate_axis_limits()
        self._initiate_plots()
        self._configure_axes()

        plt.pause(0.01)

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

        self._plot_signal_heatmap()
        self._plot_drone_frame()
        self._update_drone_points()
        plt.pause(0.01)

        if verbose:
            print(self.viewer_status_str())

        current_fps = 1.0 / self.elapsed_time if self.elapsed_time > 0.0 else 0.0
        self.fps = 0.9 * self.fps + 0.1 * current_fps
        self.last_render_time = self.time

    def viewer_status_str(self) -> str:
        return (
            f"Real time: {self.time:.2f} s, "
            f"Sim time: {self.sim.sim_time:.2f} s, "
            f"FPS: {self.fps:.2f}"
        )

    def _initiate_plots(self) -> None:
        (self.drone_points,) = self.ax1.plot([], [], "ko", ms=2.0)

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

    def _update_drone_points(self) -> None:
        self.drone_points.set_data(
            self.sim.drone_states[:, 0],
            self.sim.drone_states[:, 1],
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

        self.ax2.set_title("Single Drone Frame")
        self.ax2.set_xlabel("X (pixels)")
        self.ax2.set_ylabel("Y (pixels)")

        self.fig.tight_layout()

        if self.xlim:
            self.ax1.set_xlim(*self.xlim)

        if self.ylim:
            self.ax1.set_ylim(*self.ylim)

    def _plot_signal_heatmap(self) -> None:
        """
        Plots the 2D transmitter power heatmap in real time.
        """
        # Generate the grid for the heatmap
        xs = np.linspace(self.xlim[0], self.xlim[1], 100)
        ys = np.linspace(self.ylim[0], self.ylim[1], 100)

        # Calculate the heatmap using the simulator's tx_power_heatmap method
        heatmap = signal_strength_map(
            self.sim.drone_positions, xs, ys, f=2.4e3, mode="max"
        )

        # Plot the heatmap
        if self.signal_heatmap_image is None:
            self.signal_heatmap_image = self.ax1.imshow(
                heatmap,
                extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]],
                origin="lower",
                cmap="turbo",  # Use a visually appealing colormap
                alpha=0.7,
            )
        else:
            self.signal_heatmap_image.set_data(heatmap)

    def _plot_drone_frame(self, drone_id: int = 0) -> None:
        drone = self.sim.drones[drone_id]
        dqns = self.sim._get_drone_position_controller(drone).dqns
        signal_matrix = dqns.signal_matrix(units="dbm")
        obstacles_matrix = dqns.obstacles_matrix()
        state_frame = np.clip(signal_matrix + obstacles_matrix, 0.0, 1.0)

        if self.drone_frame_image is None:
            self.drone_frame_image = self.ax2.imshow(
                state_frame, origin="lower", cmap="gray"
            )
        else:
            self.drone_frame_image.set_data(state_frame)

    def _need_render(self) -> bool:
        if self.fps > self.max_fps:
            return False
        return self.sim.sim_time > self.time or self.fps < self.min_fps

    def _reset_timers(self) -> None:
        self.t0 = time.time()
        self.fps = 0.0
        self.last_render_time = 0.0
