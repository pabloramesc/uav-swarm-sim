"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from numpy.typing import ArrayLike

from ..simulators.evsm_simulator import EVSMSimulator
from .simple_viewer import BackgroundType, SimpleViewer


class EVSMViewer(SimpleViewer):

    def __init__(
        self,
        evsm: EVSMSimulator,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        fig_size: tuple[float, float] | None = None,
        min_fps: float = 10.0,
        max_fps: float = 60.0,
        background_type: BackgroundType = "rssi",
        show_legend: bool = False,
    ) -> None:
        super().__init__(
            sim=evsm.sim,
            limits=xlim + ylim if xlim and ylim else None,
            figsize=fig_size,
            min_fps=min_fps,
            max_fps=max_fps,
            background_type=background_type,
            show_legend=show_legend,
        )
        self.evsm = evsm

    def _update_links_lines(self) -> None:
        links_x, links_y = self._get_links_coords()
        self.spring_lines.set_data(links_x, links_y)

    def _get_links_coords(self) -> tuple[ArrayLike, ArrayLike]:
        links_x, links_y = [], []
        for drone1_idx in range(self.evsm.num_drones):

            drone1_pos = self.evsm.drone_states[drone1_idx, 0:3]
            for drone2_idx in range(self.evsm.num_drones):
                if not self.evsm.evsm_monitor.springs_matrix[drone1_idx, drone2_idx]:
                    continue

                drone2_pos = self.evsm.drone_states[drone2_idx, 0:3]
                links_x.extend([drone1_pos[0], drone2_pos[0], None])
                links_y.extend([drone1_pos[1], drone2_pos[1], None])

            if not np.any(self.evsm.evsm_monitor.springs_matrix[drone1_idx]):
                self.logger.info(f"Drone {drone1_idx} has no links.")

        return links_x, links_y
