import numpy as np

from ..environment import Environment
from ..math.distances import pairwise_self_distances
from ..math.path_loss_model import rssi_to_signal_quality, signal_strength
from ..math.connectivity import connected_clusters
from .utils import distances_to_obstacles


class RewardManager:
    """
    RewardManager computes reward signals and episode termination flags for
    Swarm DQN (SDQN) multi-agent reinforcement learning controlling UAV placement.

    Rewards consider user coverage, drone connectivity, and penalties for
    proximity to obstacles or collisions.
    """

    def __init__(self, env: Environment) -> None:
        """
        Initialize RewardManager with environment and distance thresholds.

        Parameters
        ----------
        env : Environment
            The simulation environment providing boundary, obstacles, and elevation.

        Attributes
        ----------
        d_obstacles : float
            Minimum safe distance to obstacles; below this, a small penalty applies.
        d_collision : float
            Distance threshold for collisions; below this, heavy penalty and done flag.
        """
        self.env = env
        self.d_obstacles = 10.0
        self.d_collision = 1.0

    def update(
        self, drones: np.ndarray, users: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-drone rewards and done flags for the current state.

        Parameters
        ----------
        drones : np.ndarray
            Array of shape (N, 3) with the 3D positions of N drones.
        users : np.ndarray
            Array of shape (M, 3) with the 3D positions of M users.

        Returns
        -------
        rewards : np.ndarray
            Float array of length N with each drone's reward.
        dones : np.ndarray
            Boolean array of length N indicating if the episode ends for each drone.
        """
        num_drones = drones.shape[0]

        rewards = np.zeros(num_drones)
        rewards += self.difference_coverage_rewards(drones, users)
        rewards += self.difference_connectivity_rewards(drones)

        dist = self.min_separation(drones)
        rewards[dist < self.d_obstacles] = -1.0
        rewards[dist <= self.d_collision] = -10.0

        dones = np.zeros(num_drones, dtype=bool)
        dones[dist <= self.d_collision] = True

        return rewards, dones

    def min_separation(self, drones: np.ndarray) -> np.ndarray:
        """
        Compute the minimum separation distance for each drone to any other drone or obstacle.

        Parameters
        ----------
        drones : np.ndarray
            Positions of drones, shape (N, 3).

        Returns
        -------
        np.ndarray
            Array of length N with the minimum distance to another drone or obstacle.
        """
        pairwise = pairwise_self_distances(drones)
        pairwise[pairwise <= 0.0] = np.inf
        nearest_drone = np.min(pairwise, axis=-1)

        nearest_obs = distances_to_obstacles(self.env, drones[:, :2])

        return np.minimum(nearest_drone, nearest_obs)

    def difference_coverage_rewards(
        self, drones: np.ndarray, users: np.ndarray
    ) -> np.ndarray:
        """
        Compute coverage-based difference rewards: the marginal contribution of each drone to user coverage.

        Each drone's reward = global coverage ratio - coverage ratio without that drone.

        Parameters
        ----------
        drones : np.ndarray
            Drone positions, shape (N, 3).
        users : np.ndarray
            User positions, shape (M, 3).

        Returns
        -------
        np.ndarray
            Coverage difference reward for each drone.
        """
        num_drones = drones.shape[0]
        rewards = np.zeros(num_drones)
        global_reward = self._users_coverage_ratio(drones, users)
        for i in range(num_drones):
            no_drone_reward = self._users_coverage_ratio(
                np.delete(drones, i, axis=0), users
            )
            rewards[i] = global_reward - no_drone_reward
        return rewards

    def difference_connectivity_rewards(self, drones: np.ndarray) -> np.ndarray:
        """
        Compute connectivity-based difference rewards: the marginal contribution of each drone to network connectivity.

        Each drone's reward = global connected-drones ratio - adjusted ratio without that drone.

        Parameters
        ----------
        drones : np.ndarray
            Drone positions, shape (N, 3).

        Returns
        -------
        np.ndarray
            Connectivity difference reward for each drone.
        """
        num_drones = drones.shape[0]
        rewards = np.zeros(num_drones)
        global_reward = self._connected_drones_ratio(drones)
        for i in range(num_drones):
            no_drone_reward = self._connected_drones_ratio(
                np.delete(drones, i, axis=0)
            ) * (1 - 1 / num_drones)
            rewards[i] = global_reward - no_drone_reward
        return rewards

    def _users_coverage_ratio(self, drones: np.ndarray, users: np.ndarray) -> float:
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
        return np.mean(covered)

    def _connected_drones_ratio(self, drones: np.ndarray) -> float:
        clusters = connected_clusters(drones)
        if not clusters:
            return 0.0
        largest_cluster_size = max(len(cluster) for cluster in clusters)
        return largest_cluster_size / drones.shape[0]
