import numpy as np

from simulator.environment import Environment
from simulator.math.distances import pairwise_self_distances

from .utils import (
    gaussian_decay,
    saturated_exponential,
    distances_to_obstacles,
    VisitedCells,
)


class RewardManager:

    def __init__(self, env: Environment, cell_size: float = 10.0) -> None:
        self.env = env
        self.cell_size = cell_size

        self.d_obs = 10.0
        self.d_ideal = 50.0
        self.d_max = 100.0
        self.max_links = 6

        self.w_dist = 1.0
        self.w_conn = 0.1
        self.w_expl = 1.0

        self.visited_cells = VisitedCells(cell_size=100.0)
        self.expire_time = 60.0

    def update(
        self, drone_positions: np.ndarray, time: float
    ) -> tuple[np.ndarray, np.ndarray]:

        pairwise_distances = pairwise_self_distances(drone_positions)
        pairwise_distances[pairwise_distances <= 0.0] = np.inf
        nearest_distances = np.min(pairwise_distances, axis=-1)
        connected_neighbors = np.sum(pairwise_distances < self.d_max, axis=0)

        r_dist = self.collision_rewards(drone_positions, nearest_distances)
        r_conn = self.connectivity_rewards(connected_neighbors)
        r_expl = self.exploration_rewards(drone_positions, time)

        self._update_visited_cells(drone_positions, time)

        rewards = self.w_dist * r_dist + self.w_conn * r_conn + self.w_expl * r_expl
        # rewards = np.clip(rewards, -1.0, +1.0)
        
        # dones = self.is_collision(drone_positions)
        dones = r_dist <= -0.99 # almost -1
        rewards[dones] = -10.0

        return rewards, dones

    def collision_rewards(
        self, drone_positions: np.ndarray, nearest_distances: np.ndarray
    ) -> np.ndarray:
        neighbor_repulsion_rewards = -gaussian_decay(
            nearest_distances, sigma=self.d_obs
        )

        nearest_obstacle_distances = distances_to_obstacles(
            self.env, drone_positions[:, 0:2]
        )
        osbtacle_repulsion_rewards = -gaussian_decay(
            nearest_obstacle_distances, sigma=self.d_obs
        )

        rewards = neighbor_repulsion_rewards + osbtacle_repulsion_rewards
        return np.clip(rewards, -1.0, 0.0)

    def connectivity_rewards(self, connected_neighbors: np.ndarray) -> np.ndarray:
        rewards = saturated_exponential(connected_neighbors, tau=3)
        return rewards

    def exploration_rewards(self, positions: np.ndarray, time: float) -> np.ndarray:
        last_visited_times = self.visited_cells.get_cells_time(positions)
        elapsed_times = time - last_visited_times
        elapsed_times /= self.expire_time
        return np.clip(elapsed_times, 0.0, 1.0)

    def is_collision(self, positions: np.ndarray) -> np.ndarray:
        inside = self.env.is_inside(positions)
        collision = self.env.is_collision(positions, check_altitude=False)
        return ~inside | collision

    def _get_obstacles_distances(self, positions: np.ndarray) -> np.ndarray:
        obstacles = self.env.boundary_and_obstacles
        distances = np.zeros((len(obstacles), positions.shape[0]))
        for i, obs in enumerate(obstacles):
            distances[i] = obs.distance(positions)
        return np.min()

    def _update_visited_cells(self, positions: np.ndarray, time: float) -> None:
        self.visited_cells.set_cells_time(positions, time)
