"""
MultiDroneViewer: A visualization tool for the MultiDroneSimulator.

This module provides a graphical interface to visualize the state of a swarm of drones
in a 2D environment, including their positions, links, and edge drones.
"""

import time
import numpy as np
from matplotlib import pyplot as plt
from simulator.multidrone_simulator import MultiDroneSimulator
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


class MultiDroneViewer:
    """
    A viewer for visualizing the MultiDroneSimulator in a 2D environment.

    This class provides real-time visualization of the drone swarm, including
    drone positions, links between drones, and edge drones. It also supports
    rendering boundaries and obstacles in the environment.
    """

    def __init__(
        self,
        sim: MultiDroneSimulator,
        xlim: tuple[float, float] = (-100.0, +100.0),
        ylim: tuple[float, float] = (-100.0, +100.0),
        fig_size: tuple[float, float] = (12, 6),
        min_render_steps: int = 100,
        min_render_freq: float = 1.0,
        plot_3d: bool = False,
    ):
        """
        Initializes the MultiDroneViewer.

        Parameters
        ----------
        sim : MultiDroneSimulator
            The simulator instance to visualize.
        xlim : tuple[float, float], optional
            The x-axis limits for the visualization (default is (-100.0, +100.0)).
        ylim : tuple[float, float], optional
            The y-axis limits for the visualization (default is (-100.0, +100.0)).
        fig_size : tuple[float, float], optional
            The size of the matplotlib figure (default is (12, 6)).
        min_render_steps : int, optional
            Minimum number of simulation steps between renders (default is 100).
        min_render_freq : float, optional
            Minimum rendering frequency in Hz (default is 1.0).
        plot_3d : bool, optional
            If True, enables 3D plotting (default is False).
        """
        self.sim = sim
        self.xlim = xlim
        self.ylim = ylim
        self.fig_size = fig_size
        self.plot_3d = plot_3d

        self.min_render_steps = min_render_steps
        self.min_render_freq = min_render_freq
        self.min_render_period = 1.0 / min_render_freq

        self.t0: float = None
        self.last_render_time: float = None
        self.non_render_steps: int = None

        self.fig = plt.figure(figsize=fig_size)
        if self.plot_3d:
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig.add_subplot()

        self.reset()

    def reset(self) -> None:
        """
        Resets the viewer to its initial state.

        This method clears the axes, initializes the plot elements, and resets
        the rendering timers.
        """
        self.t0 = time.time()
        self.last_render_time = 0.0
        self.non_render_steps = self.min_render_steps + 1

        self.ax.clear()

        # Use elevation map bounds if available
        if self.sim.environment.elevation is not None:
            elevation_bounds = self.sim.environment.elevation.bounds
            self.xlim = (elevation_bounds[0], elevation_bounds[2])  # xmin, xmax
            self.ylim = (elevation_bounds[1], elevation_bounds[3])  # ymin, ymax

        if self.plot_3d:
            (self.link_lines,) = self.ax.plot([], [], [], "b-", lw=0.5)
            (self.drone_points,) = self.ax.plot([], [], [], "ro", ms=2.0)
            (self.edge_drone_points,) = self.ax.plot([], [], [], "go", ms=2.0)
            
            if self.sim.environment.boundary is not None:
                boundary_coords = self.sim.environment.boundary.shape.exterior.coords.xy
                self.ax.plot(
                    boundary_coords[0],
                    boundary_coords[1],
                    [self.sim.environment.max_elevation()] * len(boundary_coords[0]),
                    "r-",
                )

            for obs in self.sim.environment.obstacles:
                obs_coords = obs.shape.exterior.coords.xy
                self.ax.plot_trisurf(
                    obs_coords[0],
                    obs_coords[1],
                    [self.sim.environment.max_elevation()] * len(obs_coords[0]),
                    color="grey",
                    alpha=0.5,
                )
                
            self.ax.set_xlim(*self.xlim)
            self.ax.set_ylim(*self.ylim)
            self.ax.set_zlim(0, self.sim.environment.max_elevation())
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

        else:
            (self.link_lines,) = self.ax.plot([], [], "b-", lw=0.5)
            (self.drone_points,) = self.ax.plot([], [], "ro", ms=2.0)
            (self.edge_drone_points,) = self.ax.plot([], [], "go", ms=2.0)

            if self.sim.environment.boundary is not None:
                self.ax.plot(
                    *self.sim.environment.boundary.shape.exterior.coords.xy, "r-"
                )

            for obs in self.sim.environment.obstacles:
                self.ax.fill(*obs.shape.exterior.coords.xy, facecolor="grey")

            self.ax.set_xlim(*self.xlim)
            self.ax.set_ylim(*self.ylim)
            self.ax.grid(True)
            self.ax.set_aspect("equal")

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
        real_time = time.time() - self.t0
        non_render_time = real_time - self.last_render_time
        if (
            force_render
            or self.sim.time > real_time
            or non_render_time >= self.min_render_period
            or self.non_render_steps >= self.min_render_steps
        ):
            self._render()
        else:
            self.non_render_steps += 1
            return

        real_time = time.time() - self.t0
        if verbose:
            fps = 1.0 / non_render_time if non_render_time > 0 else 0.0
            print(
                f"real time: {real_time:.2f} s, sim time: {self.sim.time:.2f} s, FPS: {fps:.2f}"
            )

        self.non_render_steps = 0
        self.last_render_time = real_time

    def _render(self) -> None:
        """
        Renders the current state of the simulation.

        This method updates the positions of drones, links, and edge drones
        in the visualization and redraws the plot.
        """
        if self.plot_3d:
            self._render_3d()
        else:
            self._render_2d()

    def _render_2d(self) -> None:
        """
        Renders the simulation in 2D.
        """
        self.drone_points.set_data(
            self.sim.drone_states[~self.sim.edge_drones_mask, 0],
            self.sim.drone_states[~self.sim.edge_drones_mask, 1],
        )
        
        self.edge_drone_points.set_data(
            self.sim.drone_states[self.sim.edge_drones_mask, 0],
            self.sim.drone_states[self.sim.edge_drones_mask, 1],
        )
        
        links_x, links_y = [], []
        for drone1_id in range(self.sim.num_drones):
            drone1_pos = self.sim.drone_states[drone1_id, 0:2]
            for drone2_id in range(self.sim.num_drones)[:drone1_id]:
                if not self.sim.links_matrix[drone1_id, drone2_id]:
                    continue
                drone2_pos = self.sim.drone_states[drone2_id, 0:2]
                links_x.extend([drone1_pos[0], drone2_pos[0], None])
                links_y.extend([drone1_pos[1], drone2_pos[1], None])
        self.link_lines.set_data(links_x, links_y)

        plt.pause(0.01)

    def _render_3d(self) -> None:
        """
        Renders the simulation in 3D, including drones and links.
        """
        self.drone_points._offsets3d = (
            self.sim.drone_states[~self.sim.edge_drones_mask, 0],
            self.sim.drone_states[~self.sim.edge_drones_mask, 1],
            self.sim.drone_states[~self.sim.edge_drones_mask, 2],
        )
        
        self.edge_drone_points._offsets3d = (
            self.sim.drone_states[self.sim.edge_drones_mask, 0],
            self.sim.drone_states[self.sim.edge_drones_mask, 1],
            self.sim.drone_states[self.sim.edge_drones_mask, 2],
        )

        links_x, links_y, links_z = [], [], []
        for drone1_id in range(self.sim.num_drones):
            drone1_pos = self.sim.drone_states[drone1_id, 0:3]
            for drone2_id in range(self.sim.num_drones)[:drone1_id]:
                if not self.sim.links_matrix[drone1_id, drone2_id]:
                    continue
                drone2_pos = self.sim.drone_states[drone2_id, 0:3]
                links_x.extend([drone1_pos[0], drone2_pos[0], None])
                links_y.extend([drone1_pos[1], drone2_pos[1], None])
                links_z.extend([drone1_pos[2], drone2_pos[2], None])
        self.link_lines.set_data_3d(links_x, links_y, links_z)

        plt.pause(0.01)
