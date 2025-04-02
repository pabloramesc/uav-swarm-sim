import time

import numpy as np
from matplotlib import pyplot as plt


class MultiDroneViewer:

    def __init__(
        self,
        num_drones: int,
        field_size: float = 1000.0,
        fig_size: tuple[float, float] = (12, 6),
    ):
        self.num_drones = num_drones

        self.fig = plt.figure(figsize=fig_size)
        self.ax = self.fig.add_subplot()
        (self.link_lines,) = self.ax.plot([], [], "b-", lw=0.5)
        (self.drone_points,) = self.ax.plot([], [], "ro", ms=2.0)
        self.ax.set_xlim(-field_size, field_size)
        self.ax.set_ylim(-field_size, field_size)
        self.ax.grid(True)
        self.ax.set_aspect("equal")
        plt.pause(0.01)

        self.t0 = time.time()
        self.last_render_time = 0.0
        self.non_render_steps = 0

    def update(
        self,
        links_matrix: np.ndarray,
        drone_states: np.ndarray,
        sim_time: float,
        min_render_steps: int = 100,
        min_render_freq: float = 1.0,
        force_render: bool = False,
        verbose: bool = False,
    ):
        real_time = time.time() - self.t0
        dt = real_time - self.last_render_time
        if (
            sim_time > real_time
            or self.non_render_steps >= min_render_steps
            or dt >= 1.0 / min_render_freq
            or force_render
        ):
            self.render(links_matrix, drone_states)
        else:
            self.non_render_steps += 1
            return

        real_time = time.time() - self.t0
        if verbose:
            print(f"[MultiDroneViewer]: Plot updated at {1.0/dt:.4f} FPS")

        self.non_render_steps = 0
        self.last_render_time = real_time

    def render(self, links_matrix: np.ndarray, drone_states: np.ndarray):
        links_x, links_y = [], []
        for drone1_id in range(self.num_drones):
            drone1_pos = drone_states[drone1_id, 0:2]
            for drone2_id in range(self.num_drones)[:drone1_id]:
                if not links_matrix[drone1_id, drone2_id]:
                    continue
                drone2_pos = drone_states[drone2_id, 0:2]
                links_x.extend([drone1_pos[0], drone2_pos[0], None])
                links_y.extend([drone1_pos[1], drone2_pos[1], None])

        self.link_lines.set_data(links_x, links_y)
        self.drone_points.set_data(drone_states[:, 0], drone_states[:, 1])

        plt.pause(0.01)
