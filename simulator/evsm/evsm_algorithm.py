"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.environment.environment import Environment
from simulator.math.angles import SweepAngle
from .numba_helpers import (
    obstacles_force,
    control_force,
    damping_force,
    exploration_force,
    links_matrix,
    sweep_angle,
)


class EVSM:
    """
    EVSM (Exploration and Virtual Spring Model) class for swarm behavior.

    This class models the behavior of a robot in a swarm, including control
    forces, obstacle avoidance, and exploration.
    """

    def __init__(
        self,
        env: Environment,
        ln: float = 50.0,
        ks: float = 0.2,
        kd: float = 0.8,
        d_obs: float = 10.0,
        k_obs: float = 1.0,
        k_expl: float = 0.02,
        max_force: float = 1.0,
    ) -> None:
        """
        Initializes the EVSM model with default parameters.

        Parameters
        ----------
        ln : float, optional
            Natural length of the virtual spring (default is 50.0).
        """
        self.env = env

        self.ln = ln
        self.ks = ks
        self.kd = kd
        self.d_obs = d_obs
        self.k_obs = k_obs
        self.k_expl = k_expl

        self.max_force = max_force

        self.state = np.zeros(4)  # [px, py, vx, vy]
        self.neighbors = np.zeros((0, 2))  # [px, py] of neighbors
        self.links_mask = np.zeros((0,), dtype=bool)

        self.sweep_angle: SweepAngle = None

        self.last_update_time: float = None
        self.min_update_period = 1.0
        self.exploration_force: np.ndarray = None

    @property
    def position(self) -> np.ndarray:
        """A (2,) array with the current position [px, py] of the agent in meters."""
        return self.state[0:2]

    @property
    def velocity(self) -> np.ndarray:
        """A (2,) array with the current veclocity [vx, vy] of the agent in m/s."""
        return self.state[2:4]

    def update(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        neighbors: np.ndarray,
        time: float = None,
        force_update: bool = True,
    ) -> np.ndarray:
        """
        Updates the state of the agent's position, velocity, and neighbors.

        Parameters
        ----------
        position : np.ndarray
            A (2,) array with the position [px, py] of the agent in meters.
        velocity : np.ndarray
            A (2,) array with the velocity [vx, vy] of the agent in m/s.
        neighbors : np.ndarray
            A (N,2) array with the positions [px, py] of the visible neighbors
            in meters.
        time : float, optional
            The current simulation time in seconds. Default is None.
        force_update : bool, optional
            Whether to force `links_mask` and `sweep_angle` update. Recommended
            when neighbors or environment change. Default is True.

        Returns
        -------
        np.ndarray
            Total force acting on the robot [fx, fy] in N.
        """
        self.state[0:2] = position.copy()
        self.state[2:4] = velocity.copy()
        self.neighbors = neighbors.copy()
        return self._calculate_total_force(time, force_update)

    def _calculate_total_force(
        self, time: float = None, force_update: bool = True
    ) -> np.ndarray:
        """
        Update the links mask and compute the total force acting on the agent.
        """
        if force_update or self._needs_update(time):
            self.links_mask = self._calculate_links()
            self.sweep_angle = self._calculate_sweep_angle()
            self.last_update_time = time

        damping_force = self._calculate_damping_force()

        if self.is_near_obstacle():
            obstacles_force = self._calculate_obstacles_force()
            return obstacles_force + damping_force

        control_force = self._calculate_control_force()

        if self.is_edge_robot():
            if force_update or self._needs_update(time) or self.exploration_force is None:
                exploration_force = self._calculate_exploration_force()
                self.exploration_force = self._limit_force(exploration_force)
            return control_force + damping_force + self.exploration_force

        return control_force + damping_force

    def _needs_update(self, time: float) -> bool:
        if time is None or self.last_update_time is None:
            return True
        elapsed_time = time - self.last_update_time
        return elapsed_time > self.min_update_period

    def _limit_force(self, force: np.ndarray) -> np.ndarray:
        force_mag = np.linalg.norm(force)
        if force_mag > self.max_force:
            return self.max_force * force / force_mag
        return force

    def _calculate_links(self) -> np.ndarray:
        return links_matrix(self.position, self.neighbors)

    def _calculate_control_force(self) -> np.ndarray:
        linked_neighbors = self.neighbors[self.links_mask]
        return control_force(
            self.position, linked_neighbors, ln=self.ln, ks=self.ks
        )

    def _calculate_damping_force(self) -> np.ndarray:
        return damping_force(self.velocity, kd=self.kd)

    def _calculate_exploration_force(self) -> np.ndarray:
        if not self.is_edge_robot():
            return np.zeros(2)
        region_distances, region_directions = (
            self._get_avoidance_distances_and_directions()
        )
        return exploration_force(
            region_distances,
            region_directions,
            self.sweep_angle.to_tuple(),
            ln=self.ln,
            ks=self.k_expl,
        )

    def _calculate_obstacles_force(self) -> np.ndarray:
        region_distances, region_directions = (
            self._get_avoidance_distances_and_directions()
        )
        return obstacles_force(
            region_distances, region_directions, d_min=self.d_obs, ks=self.k_obs
        )

    def _calculate_sweep_angle(self) -> SweepAngle:
        start, stop = sweep_angle(self.position, self.neighbors)
        if np.isnan((start, stop)).any():
            return None
        return SweepAngle(start, stop)

    def _get_avoidance_distances_and_directions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates distances and directions to avoidance regions.
        """
        obstacles = self.env.boundary_and_obstacles
        num_obstacles = len(obstacles)
        distances = np.zeros((num_obstacles,))
        directions = np.zeros((num_obstacles, 2))
        for i, obs in enumerate(obstacles):
            distances[i] = obs.distance(self.position)
            directions[i, :] = obs.direction(self.position)
        return distances, directions

    def is_edge_robot(self) -> bool:
        """
        Checks if the robot is at the edge of the swarm.

        Returns
        -------
        bool
            True if the robot is at the edge, False otherwise.
        """
        return self.sweep_angle is not None

    def is_near_obstacle(self) -> bool:
        """
        Checks if the robot is near any obstacle.

        Returns
        -------
        bool
            True if the robot is near an obstacle, False otherwise.
        """
        for region in self.env.boundary_and_obstacles:
            if region.distance(self.position) < self.d_obs:
                return True
        return False

    def set_natural_length(self, ln: float) -> None:
        """
        Sets the natural length of the virtual spring.

        Parameters
        ----------
        ln : float
            New natural length.

        Raises
        ------
        ValueError
            If the natural length is not greater than 0.
        """
        if ln <= 0.0:
            raise ValueError("Natural length must be greater than 0.0")
        self.ln = ln
