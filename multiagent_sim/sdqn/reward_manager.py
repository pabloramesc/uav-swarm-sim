import numpy as np

from ..environment import Environment
from ..core.metrics import area_coverage
from ..math.distances import pairwise_self_distances
from ..math.connectivity import covered_positions, globally_connected
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
        self, drones: np.ndarray, users: np.ndarray, **kwargs
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
        dones = np.zeros(num_drones, dtype=bool)

        rewards += self.difference_area_coverage_rewards(drones)
        # rewards += self.difference_users_coverage_rewards(drones, users)
        # rewards += self.difference_connectivity_rewards(drones)

        dist = self.min_separation(drones, check_drones_separation=False)
        rewards[dist < self.d_obstacles] = -1.0
        rewards[dist <= self.d_collision] = -10.0

        dones[dist <= self.d_collision] = True

        return rewards, dones

    def min_separation(
        self, drones: np.ndarray, check_drones_separation: bool = True
    ) -> np.ndarray:
        """
        Compute the minimum separation distance for each drone to the nearest obstacle, and,
        optionally, to other drones.
        """
        nearest_obs = distances_to_obstacles(self.env, drones[:, 0:2])
        
        if not check_drones_separation:
            return nearest_obs
        
        pairwise = pairwise_self_distances(drones)
        pairwise[pairwise <= 0.0] = np.inf
        nearest_drone = np.min(pairwise, axis=-1)

        return np.minimum(nearest_drone, nearest_obs)

    def difference_area_coverage_rewards(self, drones: np.ndarray) -> np.ndarray:
        """
        Compute area coverage difference rewards: the marginal contribution of each drone to
        environment area coverage.
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

    def difference_users_coverage_rewards(
        self, drones: np.ndarray, users: np.ndarray
    ) -> np.ndarray:
        """
        Compute coverage-based difference rewards: the marginal contribution of each drone to user coverage.
        """
        num_drones = drones.shape[0]
        rewards = np.zeros(num_drones)
        global_reward = self._coverage_ratio(tx_positions=drones, rx_positions=users)
        for i in range(num_drones):
            no_drone_reward = self._coverage_ratio(
                tx_positions=np.delete(drones, i, axis=0), rx_positions=users
            )
            rewards[i] = global_reward - no_drone_reward
        return rewards

    def difference_connectivity_rewards(self, drones: np.ndarray) -> np.ndarray:
        """
        Compute connectivity-based difference rewards: the marginal contribution of each drone to network connectivity.
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
        covered = covered_positions(tx_positions, rx_positions)
        return len(covered) / max(len(rx_positions), 1)

    def _global_connections(self, positions: np.ndarray):
        connected = globally_connected(positions)
        return len(connected) / max(len(positions), 1)
