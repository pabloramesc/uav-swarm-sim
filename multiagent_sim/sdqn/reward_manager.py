import numpy as np

from ..environment import Environment
from ..math.distances import pairwise_self_distances
from ..math.path_loss_model import rssi_to_signal_quality, signal_strength
from .utils import (
    VisitedCells,
    distances_to_obstacles,
    gaussian_decay,
    saturated_exponential,
)


class RewardManager:

    def __init__(self, env: Environment) -> None:
        self.env = env

        self.d_obs = 10.0
        self.d_ideal = 50.0
        self.d_max = 100.0
        self.max_links = 6

        self.w_coll = 1.0
        self.w_conn = 0.0
        self.w_covr = 0.1
        self.w_expl = 0.0

        self.visited_cells = VisitedCells(cell_size=50.0)
        self.expire_time = 60.0
        self.min_quality = 0.0

    def update(
        self, drones: np.ndarray, users: np.ndarray, time: float
    ) -> tuple[np.ndarray, np.ndarray]:

        # r_coll = self.collision_rewards(drones)
        # r_conn = self.connectivity_rewards(drones)
        # r_covr = self.users_coverage_reward(drones, users)
        # r_expl = 0.0

        # # self._update_visited_cells(drones, time)

        # rewards = (
        #     self.w_coll * r_coll
        #     + self.w_conn * r_conn
        #     + self.w_covr * r_covr
        #     + self.w_expl * r_expl
        # )
        # rewards = np.clip(rewards, -1.0, +1.0)

        # # dones = self.is_collision(drone_positions)
        # dones = r_coll <= -0.9  # almost -1
        # rewards[dones] = -10.0

        num_drones = drones.shape[0]
        rewards = np.zeros(num_drones)
        dones = np.zeros(num_drones, dtype=bool)

        # ratio = self.users_coverage_ratio(drones, users)
        # rewards[:] = ratio**2

        # quality = self.signal_quality(drones)
        # rewards[quality < self.min_quality] = -1
        # rewards = (1.0 - quality)**2
        
        rewards = self.coverage_reward(drones, users)

        dist = self.min_separation(drones)
        rewards[dist < self.d_obs] = -1

        rewards[dist <= 0.0] = -10
        dones[dist <= 0.0] = True

        return rewards, dones
    
    def coverage_reward(self, drones: np.ndarray, users: np.ndarray) -> np.ndarray:
        num_drones = drones.shape[0]
        rewards = np.zeros(num_drones)
        for i in range(num_drones):
            drones_rssi = signal_strength(
                tx_positions=np.delete(drones, i, axis=0),
                rx_positions=drones[i],
                f=2412,
                n=2.4,
                tx_power=20.0,
                mode="max",
            )
            drones_quality = rssi_to_signal_quality(drones_rssi, vmin=-80.0, vmax=-30.0)
            users_rssi = signal_strength(
                tx_positions=users,
                rx_positions=drones[i],
                f=2412,
                n=2.4,
                tx_power=20.0,
                mode="max",
            )
            users_quality = rssi_to_signal_quality(users_rssi, vmin=-80.0, vmax=-30.0)
            rewards[i] = users_quality - drones_quality
        return rewards

    def min_separation(self, drones: np.ndarray) -> np.ndarray:
        pairwise = pairwise_self_distances(drones)
        pairwise[pairwise <= 0.0] = np.inf
        nearest_drone = np.min(pairwise, axis=-1)

        nearest_obs = distances_to_obstacles(self.env, drones[:, :2])

        return np.minimum(nearest_drone, nearest_obs)

    def signal_quality(self, drones: np.ndarray) -> np.ndarray:
        n = drones.shape[0]
        quality = np.zeros(n)

        for i in range(n):
            rssi = signal_strength(
                tx_positions=np.delete(drones, i, axis=0),
                rx_positions=drones[i],
                f=2412,
                n=2.4,
                tx_power=20.0,
                mode="max",
            )
            quality[i] = rssi_to_signal_quality(rssi, vmin=-80.0)

        return quality

    # def collision_rewards(self, drones: np.ndarray) -> np.ndarray:
    #     pairwise_distances = pairwise_self_distances(drones)
    #     pairwise_distances[pairwise_distances <= 0.0] = np.inf
    #     nearest_distances = np.min(pairwise_distances, axis=-1)
    #     neighbor_repulsion_rewards = -gaussian_decay(
    #         nearest_distances, sigma=self.d_obs
    #     )

    #     nearest_obstacle_distances = distances_to_obstacles(self.env, drones[:, 0:2])
    #     osbtacle_repulsion_rewards = -gaussian_decay(
    #         nearest_obstacle_distances, sigma=self.d_obs
    #     )

    #     rewards = neighbor_repulsion_rewards + osbtacle_repulsion_rewards
    #     return np.clip(rewards, -1.0, 0.0)

    # def connectivity_rewards(self, drones: np.ndarray) -> np.ndarray:
    #     num_drones = drones.shape[0]
    #     rewards = np.zeros(num_drones)
    #     for i in range(num_drones):
    #         rssi = signal_strength(
    #             tx_positions=np.delete(drones, i, axis=0),
    #             rx_positions=drones[i],
    #             f=2412,
    #             n=2.4,
    #             tx_power=20.0,
    #             mode="max",
    #         )
    #         quality = rssi_to_signal_quality(rssi, vmin=-80.0)
    #         rewards[i] = 1.0 - quality if quality > self.min_quality else -1.0
    #     return rewards

    def users_coverage_ratio(self, drones: np.ndarray, users: np.ndarray) -> float:
        rssi = signal_strength(
            tx_positions=drones,
            rx_positions=users,
            f=2412,
            n=2.4,
            tx_power=20.0,
            mode="max",
        )
        quality = rssi_to_signal_quality(rssi, vmin=-80.0)
        covered = quality > self.min_quality
        return np.sum(covered) / users.shape[0]

    # def exploration_rewards(self, positions: np.ndarray, time: float) -> np.ndarray:
    #     last_visited_times = self.visited_cells.get_cells_time(positions)
    #     elapsed_times = time - last_visited_times
    #     elapsed_times /= self.expire_time
    #     return np.clip(elapsed_times, 0.0, 1.0)

    # def coverage_rewards(self, positions: np.ndarray) -> np.ndarray:
    #     rssi = np.zeros(positions.shape[0])
    #     for i, pos in enumerate(positions):
    #         neighbor_positions = np.delete(positions, i, axis=0)
    #         rssi[i] = signal_strength(
    #             tx_positions=neighbor_positions, rx_positions=pos, f=2.4e3, mode="max"
    #         )
    #     quality = rssi_to_signal_quality(rssi)
    #     rewards = (1.0 - quality / 100.0) ** 10
    #     return rewards

    # def distance_rewards(self, nearest_distances: np.ndarray) -> np.ndarray:
    #     rewards = 1.0 - gaussian_decay(nearest_distances, sigma=self.d_max)
    #     return np.clip(rewards, 0.0, 1.0)

    # def is_collision(self, positions: np.ndarray) -> np.ndarray:
    #     inside = self.env.is_inside(positions)
    #     collision = self.env.is_collision(positions, check_altitude=False)
    #     return ~inside | collision

    # def _get_obstacles_distances(self, positions: np.ndarray) -> np.ndarray:
    #     obstacles = self.env.boundary_and_obstacles
    #     distances = np.zeros((len(obstacles), positions.shape[0]))
    #     for i, obs in enumerate(obstacles):
    #         distances[i] = obs.distance(positions)
    #     return np.min(distances, axis=0)

    # def _update_visited_cells(self, positions: np.ndarray, time: float) -> None:
    #     self.visited_cells.set_cells_time(positions, time)
