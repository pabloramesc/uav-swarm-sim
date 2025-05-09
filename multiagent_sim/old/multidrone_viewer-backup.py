"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

from simulator.multidrone_simulator_evsm import MultiDroneSimulatorEVSM

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3D, Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D


class MultiDroneViewer:
    """
    A viewer for visualizing the MultiDroneSimulator in a 2D environment.

    This class provides real-time visualization of the drone swarm, including
    drone positions, links between drones, and edge drones. It also supports
    rendering boundaries and obstacles in the environment.
    """

    def __init__(
        self,
        sim: MultiDroneSimulatorEVSM,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        zlim: tuple[float, float] = None,
        fig_size: tuple[float, float] = None,
        min_fps: float = 10.0,
        is_3d: bool = False,
    ):
        self.sim = sim
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.min_fps = min_fps
        self.is_3d = is_3d

        self.t0: float = None
        self.last_render_time: float = None

        self.link_lines: Line2D | Line3DCollection = None
        self.drone_points: Line2D | Line3D = None
        self.edge_drone_points: Line2D | Line3D = None

        self.fig = plt.figure(figsize=fig_size)
        self.ax: Axes | Axes3D = None
        if self.is_3d:
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig.add_subplot()

        self.reset()

    @property
    def time(self) -> float:
        return time.time() - self.t0

    @property
    def time_since_render(self) -> float:
        return self.time - self.last_render_time

    def reset(self) -> None:
        """
        Resets the viewer to its initial state.

        This method clears the axes, initializes the plot elements, and resets
        the rendering timers.
        """
        self.ax.clear()
        self._reset_timers()
        self._set_axis_limits()
        self._initiate_plots()
        self._plot_elevation_map()
        self._configure_axis()
        plt.pause(0.01)

    def update(
        self,
        force: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Updates the viewer based on the current simulation state.

        This method decides whether to render the visualization based on the
        elapsed time, simulation time, and rendering constraints.

        Parameters
        ----------
        force : bool, optional
            If True, forces rendering regardless of constraints (default is False).
        verbose : bool, optional
            If True, prints real-time and simulation-time statistics (default is False).
        """
        if not (force or self._need_render()):
            return

        self._render()

        if verbose:
            self._print_fps()

        self.last_render_time = self.time

    def _initiate_plots(self) -> None:
        if self.is_3d:
            self._initiate_plots_3d()
        else:
            self._initiate_plots_2d()

    def _initiate_plots_2d(self) -> None:
        (self.link_lines,) = self.ax.plot([], [], "b-", lw=0.5)
        (self.drone_points,) = self.ax.plot([], [], "ro", ms=2.0)
        (self.edge_drone_points,) = self.ax.plot([], [], "go", ms=2.0)

        if self.sim.environment.boundary is not None:
            self.ax.plot(*self.sim.environment.boundary.shape.exterior.coords.xy, "r-")

        for obs in self.sim.environment.obstacles:
            self.ax.fill(*obs.shape.exterior.coords.xy, facecolor="grey")

    def _initiate_plots_3d(self) -> None:
        # (self.link_lines,) = self.ax.plot([], [], [], "b-", lw=0.5)
        self.link_lines = Line3DCollection([], colors="blue", linewidth=0.5)
        self.ax.add_collection3d(self.link_lines)
        (self.drone_points,) = self.ax.plot([], [], [], "ro", ms=2.0)
        (self.edge_drone_points,) = self.ax.plot([], [], [], "go", ms=2.0)

        if self.sim.environment.boundary is not None:
            coords = np.array(self.sim.environment.boundary.shape.exterior.coords)
            faces = self._get_polygon_faces(coords)
            poly = Poly3DCollection(faces, alpha=0.5, facecolor="r", edgecolor="k")
            self.ax.add_collection3d(poly)

        for obs in self.sim.environment.obstacles:
            coords = np.array(obs.shape.exterior.coords)
            faces = self._get_polygon_faces(coords, closed=True)
            poly = Poly3DCollection(faces, alpha=0.5, facecolor="gray", edgecolor="k")
            self.ax.add_collection3d(poly)

    def _get_polygon_faces(
        self, coords: np.ndarray, closed: bool = False
    ) -> np.ndarray:
        faces = []
        for i in range(len(coords) - 1):
            face = np.zeros((4, 3))
            face[:, 0:2] = coords[[i, i, i + 1, i + 1], :]
            face[:, 2] = np.array(self.zlim + self.zlim[::-1])
            faces.append(face)
        if closed:
            face = np.zeros((coords.shape[0], 3))
            face[:, 0:2] = coords
            face[:, 2] = self.zlim[1] * np.ones(coords.shape[0])
            faces.append(face)
        return faces

    def _render(self) -> None:
        """
        Renders the simulation in 2D.
        """
        self._set_drones_data()
        self._set_links_data()
        plt.pause(0.01)

    def _set_drones_data(self) -> None:
        self.drone_points.set_data(
            self.sim.drone_states[~self.sim.edge_drones_mask, 0],
            self.sim.drone_states[~self.sim.edge_drones_mask, 1],
        )
        self.edge_drone_points.set_data(
            self.sim.drone_states[self.sim.edge_drones_mask, 0],
            self.sim.drone_states[self.sim.edge_drones_mask, 1],
        )
        if self.is_3d:
            self.drone_points.set_3d_properties(
                self.sim.drone_states[~self.sim.edge_drones_mask, 2]
            )
            self.edge_drone_points.set_3d_properties(
                self.sim.drone_states[self.sim.edge_drones_mask, 2]
            )

    def _set_links_data(self) -> None:
        links_x, links_y, links_z = self._get_links_coords()
        if self.is_3d:
            xyz = np.array([links_x, links_y, links_z], dtype=np.float32).T
            num_lines = xyz.shape[0] // 3
            segments = [xyz[i * 3 : i * 3 + 2] for i in range(num_lines)]
            self.link_lines.set_segments(segments)
        else:
            self.link_lines.set_data(links_x, links_y)
        

    def _set_axis_limits(self) -> None:
        if self.sim.environment.elevation_map is None:
            self.zlim = (0.0, 100.0)
            return
        bounds = self.sim.environment.elevation_map.bounds
        south_west = self.sim.environment.geo2enu((bounds.bottom, bounds.left, 0.0))
        north_east = self.sim.environment.geo2enu((bounds.top, bounds.right, 0.0))
        self.xlim = (south_west[0], north_east[0])
        self.ylim = (south_west[1], north_east[1])
        if self.is_3d:
            self.zlim = (
                self.sim.environment.elevation_map.min_elevation,
                self.sim.environment.elevation_map.max_elevation,
            )

    def _plot_elevation_map(self) -> None:
        if self.sim.environment.elevation_map is None:
            return
        self.ax.imshow(
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

    def _get_links_coords(self) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        links_x, links_y, links_z = [], [], []
        for drone1_idx in range(self.sim.num_drones):
            drone1_pos = self.sim.drone_states[drone1_idx, 0:3]
            for drone2_idx in range(self.sim.num_drones)[:drone1_idx]:
                if not self.sim.links_matrix[drone1_idx, drone2_idx]:
                    continue
                drone2_pos = self.sim.drone_states[drone2_idx, 0:3]
                links_x.extend([drone1_pos[0], drone2_pos[0], None])
                links_y.extend([drone1_pos[1], drone2_pos[1], None])
                if self.is_3d:
                    links_z.extend([drone1_pos[2], drone2_pos[2], None])
        return links_x, links_y, links_z

    def _configure_axis(self) -> None:
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        self.fig.tight_layout()

        if self.xlim is not None:
            self.ax.set_xlim(*self.xlim)

        if self.ylim is not None:
            self.ax.set_ylim(*self.ylim)

        if not self.is_3d:
            return

        self.ax.set_ylabel("Z (m)")

        if self.zlim is not None:
            self.ax.set_zlim(*self.zlim)

        if self.xlim is not None and self.ylim is not None and self.zlim is not None:
            self.ax.set_box_aspect(
                (np.ptp(self.xlim), np.ptp(self.ylim), np.ptp(self.zlim))
            )

    def _need_render(self) -> bool:
        min_render_period = 1.0 / self.min_fps
        return self.sim.time >= self.time or self.time_since_render >= min_render_period

    def _reset_timers(self) -> None:
        self.t0 = time.time()
        self.last_render_time = 0.0

    def _print_fps(self) -> None:
        fps = 1.0 / self.time_since_render if self.time_since_render > 0.0 else 0.0
        print(
            f"real time: {self.time:.2f} s, sim time: {self.sim.time:.2f} s, FPS: {fps:.2f}"
        )
