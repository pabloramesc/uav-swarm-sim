"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from numpy.typing import ArrayLike

from ..core.evsm_simulator import EVSMSimulator
from .simple_viewer import BackgroundType, SimpleViewer


class EVSMViewer(SimpleViewer):

    def __init__(
        self,
        sim: EVSMSimulator,
        xlim: tuple[float, float] = None,
        ylim: tuple[float, float] = None,
        fig_size: tuple[float, float] = None,
        min_fps: float = 10.0,
        max_fps: float = 60.0,
        background_type: BackgroundType = "rssi",
    ) -> None:
        super().__init__(sim, xlim, ylim, fig_size, min_fps, max_fps, background_type)
        self.sim: EVSMSimulator = sim

    def _init_plots(self) -> None:
        """
        Initiate spring lines, edge drones, and SimpleViewer plots.
        """
        (self.spring_lines,) = self.ax.plot([], [], "g-", lw=0.5, label="springs")
        (self.edge_drone_points,) = self.ax.plot([], [], "rx", label="edge drones")
        super()._init_plots()

    def _update_plots(self) -> None:
        super()._update_plots()
        self._update_links_lines()

    def _update_agent_points(self) -> None:
        super()._update_agent_points()
        self.drone_points.set_data(
            self.sim.drone_states[~self.sim.evsm_monitor.edge_mask, 0],
            self.sim.drone_states[~self.sim.evsm_monitor.edge_mask, 1],
        )
        self.edge_drone_points.set_data(
            self.sim.drone_states[self.sim.evsm_monitor.edge_mask, 0],
            self.sim.drone_states[self.sim.evsm_monitor.edge_mask, 1],
        )

    def _update_links_lines(self) -> None:
        links_x, links_y = self._get_links_coords()
        self.spring_lines.set_data(links_x, links_y)

    def _get_links_coords(self) -> tuple[ArrayLike, ArrayLike]:
        links_x, links_y = [], []
        for drone1_idx in range(self.sim.num_drones):

            drone1_pos = self.sim.drone_states[drone1_idx, 0:3]
            for drone2_idx in range(self.sim.num_drones):
                if not self.sim.evsm_monitor.springs_matrix[drone1_idx, drone2_idx]:
                    continue

                drone2_pos = self.sim.drone_states[drone2_idx, 0:3]
                links_x.extend([drone1_pos[0], drone2_pos[0], None])
                links_y.extend([drone1_pos[1], drone2_pos[1], None])

            if not np.any(self.sim.evsm_monitor.springs_matrix[drone1_idx]):
                self.logger.info(f"Drone {drone1_idx} has no links.")

        return links_x, links_y
