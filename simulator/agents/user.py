"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.agents import Agent
from simulator.environment import Environment


class User(Agent):
    """
    Represents a user agent in the simulation environment.

    The user agent performs a random walk within the environment.
    """

    def __init__(self, id: int, env: Environment):
        """
        Initializes the user agent with a unique ID, maximum speed, and maximum acceleration.

        Parameters
        ----------
        id : int
            Unique identifier for the user agent.
        env : Environment
            The simulation environment the agent interacts with.
        """
        super().__init__(id=id, type="user", env=env)
        self.max_speed = 1.0  # Maximum horizontal speed in m/s
        self.max_climb_rate = 0.2  # Maximum ascent/descent rate in m/s
        self.turning_rate = 0.3  # Controls how quickly the agent can change direction

    def update(self, dt: float = 0.01) -> None:
        """
        Updates the state of the user agent by performing a random walk.

        Parameters
        ----------
        dt : float, optional
            The time step in seconds (default is 0.01).
        """
        super().update(dt)
        self.random_walk(dt)

    def random_walk(self, dt: float = 0.01) -> None:
        """
        Performs a random walk by generating random velocity changes and avoiding obstacles.

        The random walk is constrained by the maximum speed and incorporates smooth
        direction changes and obstacle avoidance.

        Parameters
        ----------
        dt : float, optional
            The time step in seconds (default is 0.01).
        """
        # Generate a random velocity change
        random_direction = np.random.uniform(-1, 1, size=2)  # Random 2D direction
        random_direction /= np.linalg.norm(random_direction)  # Normalize direction
        random_velocity = random_direction * np.random.uniform(
            0.0, self.max_speed
        )  # Scale by max speed

        # Smoothly blend current velocity with the random velocity
        smooth_velocity = (1 - self.turning_rate) * self.state[3:5] + (
            self.turning_rate * random_velocity
        )

        # Add obstacle avoidance force
        avoidance_force = self._calculate_obstacle_avoidance()
        total_velocity = smooth_velocity + avoidance_force * dt

        # Limit the horizontal velocity to the maximum speed
        horizontal_speed = np.linalg.norm(total_velocity)
        if horizontal_speed > self.max_speed:
            total_velocity *= self.max_speed / horizontal_speed

        # Update horizontal velocity and position
        self.state[3:5] = total_velocity
        self.state[0:2] += self.state[3:5] * dt

        # Adjust altitude toward the target elevation
        target_altitude = self.environment.get_elevation(
            self.state[0:2]
        )  # Get target elevation
        altitude_error = (
            target_altitude - self.state[2]
        )  # Difference between target and current altitude

        # Proportional control for vertical velocity
        vertical_velocity = np.clip(
            altitude_error, -self.max_climb_rate, +self.max_climb_rate
        )
        self.state[5] = vertical_velocity  # Update vertical velocity

        # Update vertical position
        self.state[2] += self.state[5] * dt

    def _calculate_obstacle_avoidance(self) -> np.ndarray:
        """
        Calculates a repulsion force to avoid nearby obstacles.

        Returns
        -------
        np.ndarray
            A (2,) array representing the horizontal repulsion force [fx, fy].
        """
        repulsion_force = np.zeros(2)
        repulsion_radius = 5.0  # Distance within which obstacles influence the agent
        repulsion_strength = 2.0  # Strength of the repulsion force

        num_obstacles = len(self.environment.obstacles)
        distances = np.zeros((num_obstacles, 1))
        directions = np.zeros((num_obstacles, 2))
        for i, obstacle in enumerate(self.environment.obstacles):
            distances[i] = obstacle.distance(self.position[0:2])
            directions[i] = obstacle.direction(self.position[0:2])

        is_near = distances < repulsion_radius
        forces = repulsion_strength * distances * (-directions) * is_near
        repulsion_force = np.sum(forces, axis=0)
        return repulsion_force
