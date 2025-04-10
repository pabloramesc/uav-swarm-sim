import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from simulator.multidrone_simulator import MultiDroneSimulator


class MultiDroneViewer:

    def __init__(
        self,
        sim: MultiDroneSimulator,
        xlim: tuple[float, float] = (-100.0, +100.0),
        ylim: tuple[float, float] = (-100.0, +100.0),
        fig_size: tuple[float, float] = (12, 6),
        min_render_steps: int = 100,
        min_render_freq: float = 1.0,
    ):
        self.sim = sim
        self.xlim = xlim
        self.ylim = ylim
        self.fig_size = fig_size
        
        self.min_render_steps = min_render_steps
        self.min_render_freq = min_render_freq
        self.min_render_period = 1.0 / min_render_freq

        self.t0: float = None
        self.last_render_time: float = None
        self.non_render_steps: int = None

        self.fig = plt.figure(figsize=fig_size)
        self.ax = self.fig.add_subplot()
        
        self.reset()
        
    def reset(self) -> None:
        self.t0 = time.time()
        self.last_render_time = 0.0
        self.non_render_steps = self.min_render_steps + 1
        
        self.ax.clear()

        (self.link_lines,) = self.ax.plot([], [], "b-", lw=0.5)
        (self.drone_points,) = self.ax.plot([], [], "ro", ms=2.0)
        (self.edge_drone_points,) = self.ax.plot([], [], "go", ms=2.0)

        if self.sim.boundary is not None:
            self.ax.plot(*self.sim.boundary.shape.exterior.coords.xy, "k-")
        for obs in self.sim.obstacles:
            self.ax.fill(
                *obs.shape.exterior.coords.xy, edgecolor="black", facecolor="grey"
            )

        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.grid(True)
        self.ax.set_aspect("equal")

        plt.pause(0.01)

    def update(
        self,
        force_render: bool = False,
        verbose: bool = False,
    ):
        real_time = time.time() - self.t0
        dt = real_time - self.last_render_time
        if (
            self.sim.time > real_time
            or self.non_render_steps >= self.min_render_steps
            or dt >= self.min_render_period
            or force_render
        ):
            self.render()
        else:
            self.non_render_steps += 1
            return

        real_time = time.time() - self.t0
        if verbose:
            fps = 1.0 / dt if dt > 0 else 0.0
            print(f"real time: {real_time:.2f} s, sim time: {self.sim.time:.2f} s, FPS: {fps:.2f}")

        self.non_render_steps = 0
        self.last_render_time = real_time

    def render(self):
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

        self.edge_drone_points.set_data(
            self.sim.drone_states[self.sim.edge_drones_mask, 0],
            self.sim.drone_states[self.sim.edge_drones_mask, 1],
        )
        self.drone_points.set_data(
            self.sim.drone_states[~self.sim.edge_drones_mask, 0],
            self.sim.drone_states[~self.sim.edge_drones_mask, 1],
        )

        plt.pause(0.01)
