import numpy as np
from numpy.typing import NDArray

from ..environment import Environment
from ..simulators.metrics import area_coverage
from ..math.distances import pairwise_self_distances
from ..math.coverage import covered_positions, coverage_matrix
from ..math.connectivity import globally_connected
from .utils import distances_to_obstacles


class RewardManager:
    """
    Computes reward signals and episode termination flags for Swarm DQN (SDQN)
    multi-agent reinforcement learning controlling UAV placement.

    Rewards consider user coverage, network connectivity, and penalties for
    proximity to obstacles or collisions.
    """

    def __init__(
        self, env: Environment, obstacle_dist: float = 10.0, collision_dist: float = 0.0
    ) -> None:
        """
        Initialize RewardManager with environment and distance thresholds.

        Args:
            env (Environment): Simulation environment with boundaries, obstacles, and elevation.
            obstacle_dist (float): Minimum safe distance to obstacles; small penalty below this.
            collision_dist (float): Distance threshold for collisions; heavy penalty and episode termination.
        """
        self.env = env
        self.d_obstacles = obstacle_dist
        self.d_collision = collision_dist

    def compute_rewards(
        self, drones: NDArray[np.floating], users: NDArray[np.floating], **kwargs
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
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
        rewards = np.zeros(num_drones, dtype=np.float32)
        dones = np.zeros(num_drones, dtype=np.bool_)

        # rewards += self.difference_area_coverage_rewards(drones)
        rewards += self.users_coverage_difference_rewards(drones, users)
        # rewards += self.users_coverage_fractional_reward(drones, users)
        # rewards += self.difference_connectivity_rewards(drones)

        dist = self.min_separation(drones, check_drones_separation=True)
        rewards[dist <= self.d_obstacles] = -1.0

        collided = dist <= self.d_collision
        dones[collided] = True
        rewards[collided] = -10.0

        return rewards, dones

    def min_separation(
        self, drones: np.ndarray, check_drones_separation: bool = True
    ) -> np.ndarray:
        """Compute the minimum separation distance for each drone to the nearest
        obstacle, and, optionally, to other drones.
        """
        nearest_obs = distances_to_obstacles(self.env, drones[:, 0:2])

        if not check_drones_separation:
            return nearest_obs

        pairwise = pairwise_self_distances(drones)
        pairwise[pairwise <= 0.0] = np.inf
        nearest_drone = np.min(pairwise, axis=-1)

        return np.minimum(nearest_drone, nearest_obs)

    def users_coverage_difference_rewards(
        self, drones: np.ndarray, users: np.ndarray
    ) -> NDArray[np.float32]:
        """Compute difference rewards for user coverage.

        The marginal contribution of each drone to user coverage calculated as
        the difference between the total coverage with all drones and the
        coverage ratio if that drone were removed.
        """
        num_drones = drones.shape[0]
        rewards = np.zeros(num_drones, dtype=np.float32)
        global_reward = self._coverage_ratio(tx_positions=drones, rx_positions=users)
        for i in range(num_drones):
            no_drone_reward = self._coverage_ratio(
                tx_positions=np.delete(drones, i, axis=0), rx_positions=users
            )
            rewards[i] = global_reward - no_drone_reward
        return rewards

    def users_coverage_fractional_rewards(
        self, drones: np.ndarray, users: np.ndarray
    ) -> NDArray[np.float32]:
        """Compute fractional rewards for user coverage.

        Each drone gets fractional credit for users it covers,
        divided by number of drones covering the same user.
        """
        coverage = coverage_matrix(drones, users)  # (N, M) boolean
        users_covered_count = np.sum(coverage, axis=0)  # (M,)
        users_covered_count[users_covered_count == 0] = 1  # avoid division by zero
        fractional_credit = coverage / users_covered_count  # broadcasting
        rewards = np.sum(fractional_credit, axis=1)  # sum over users for each drone
        rewards = rewards / users.shape[0]  # normalize to the number of users ratio
        return rewards

    def area_coverage_difference_rewards(self, drones: np.ndarray) -> np.ndarray:
        """Compute area coverage difference rewards: the marginal contribution
        of each drone to environment area coverage.
        """
        num_drones = drones.shape[0]
        rewards = np.zeros(num_drones)
        global_reward = area_coverage(env=self.env, tx_positions=drones)
        for i in range(num_drones):
            no_drone_reward = area_coverage(
                env=self.env, tx_positions=np.delete(drones, i, axis=0)
            )
            rewards[i] = global_reward - no_drone_reward
        return rewards

    def connectivity_difference_rewards(self, drones: np.ndarray) -> np.ndarray:
        """Compute difference rewards: the marginal
        contribution of each drone to network connectivity.
        """
        num_drones = drones.shape[0]
        rewards = np.zeros(num_drones)
        global_reward = self._global_connections(positions=drones)
        for i in range(num_drones):
            no_drone_reward = self._global_connections(
                positions=np.delete(drones, i, axis=0)
            ) * (1 - 1 / num_drones)
            rewards[i] = global_reward - no_drone_reward
        return rewards

    def _coverage_ratio(self, tx_positions: np.ndarray, rx_positions: np.ndarray):
        covered_mask = covered_positions(tx_positions, rx_positions)
        return np.sum(covered_mask) / max(len(rx_positions), 1)

    def _global_connections(self, positions: np.ndarray):
        connected_idx = globally_connected(positions)
        return len(connected_idx) / max(len(positions), 1)
