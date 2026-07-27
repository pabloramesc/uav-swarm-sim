import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..environment import Environment
from ..math.coverage import coverage_matrix, covered_positions
from ..math.distances import pairwise_self_distances

RewardType = Literal["global", "fractional", "difference"]


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for RewardManager."""

    # Distances
    collision_dist: float = 0.0

    # Reward types
    users_coverage: RewardType | None = "global"

    # Weights / constants
    weight_users_coverage: float = 1.0
    collision_penalty: float = -1.0

    def __post_init__(self) -> None:
        if self.users_coverage not in (None, "global", "fractional", "difference"):
            raise ValueError(f"Unknown user-coverage reward: {self.users_coverage!r}.")
        if not math.isfinite(self.collision_dist) or self.collision_dist < 0.0:
            raise ValueError("collision_dist must be non-negative and finite.")
        if not math.isfinite(self.weight_users_coverage):
            raise ValueError("weight_users_coverage must be finite.")
        if not math.isfinite(self.collision_penalty):
            raise ValueError("collision_penalty must be finite.")


class RewardManager:
    """
    Computes reward signals and episode termination flags for Swarm DQN (SDQN)
    multi-agent reinforcement learning controlling UAV placement.

    Rewards combine user coverage with penalties for invalid proximity to
    boundaries, obstacles, or other drones.
    """

    def __init__(self, env: Environment, config: RewardConfig) -> None:
        """
        Initialize RewardManager with environment and reward configuration.

        Args:
            env: Simulation environment.
            config: RewardConfig instance with distances, reward types and weights.
        """
        self.env = env
        self.config = config

    def compute_rewards(
        self, drones: NDArray[np.floating], users: NDArray[np.floating]
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

        cfg = self.config

        if cfg.users_coverage == "global":
            users_coverage_rewards = self.users_coverage_global_rewards(drones, users)
        elif cfg.users_coverage == "fractional":
            users_coverage_rewards = self.users_coverage_fractional_rewards(
                drones, users
            )
        elif cfg.users_coverage == "difference":
            users_coverage_rewards = self.users_coverage_difference_rewards(
                drones, users
            )
        else:
            users_coverage_rewards = np.zeros(num_drones, dtype=np.float32)

        rewards += cfg.weight_users_coverage * users_coverage_rewards

        dist = self.min_separation(drones, check_drones_separation=True)
        collided = dist <= cfg.collision_dist
        rewards[collided] = cfg.collision_penalty
        dones[collided] = True

        return rewards, dones

    def min_separation(
        self, drones: np.ndarray, check_drones_separation: bool = True
    ) -> np.ndarray:
        """Compute the minimum separation distance for each drone to the nearest
        obstacle, and, optionally, to other drones.
        """
        num_drones = len(drones)
        if num_drones == 0:
            return np.zeros(0, dtype=np.float64)

        obstacles = self.env.boundary_and_obstacles
        obstacle_distances = np.stack(
            [obstacle.distance(drones[:, :2]) for obstacle in obstacles]
        )
        nearest_obs = np.min(obstacle_distances, axis=0)

        if not check_drones_separation or num_drones == 1:
            return nearest_obs

        pairwise = pairwise_self_distances(drones)
        np.fill_diagonal(pairwise, np.inf)
        nearest_drone = np.min(pairwise, axis=-1)

        return np.minimum(nearest_drone, nearest_obs)

    def users_coverage_global_rewards(
        self, drones: np.ndarray, users: np.ndarray
    ) -> NDArray[np.float32]:
        """Compute global rewards for user coverage.

        Each drone gets the same reward equal to the fraction of users covered
        by at least one drone.
        """
        coverage = covered_positions(drones, users)  # boolean array of length M
        reward = np.sum(coverage) / max(len(users), 1)  # avoid division by zero
        rewards = np.full(drones.shape[0], reward, dtype=np.float32)
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
        if users.shape[0] == 0:
            return np.zeros(drones.shape[0], dtype=np.float32)
        rewards = rewards / users.shape[0]  # normalize to the number of users ratio
        return rewards.astype(np.float32, copy=False)

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

    def _coverage_ratio(self, tx_positions: np.ndarray, rx_positions: np.ndarray):
        covered_mask = covered_positions(tx_positions, rx_positions)
        return np.sum(covered_mask) / max(len(rx_positions), 1)
