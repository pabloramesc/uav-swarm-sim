"""
Abstract base class for exploration policies.
"""

from abc import ABC, abstractmethod
from typing import Literal
import numpy as np


class ExplorationPolicy(ABC):
    """
    Base class for exploration policies.

    Defines the interface for exploration policies used in reinforcement learning.
    """

    @abstractmethod
    def select_action(self, q_values: np.ndarray) -> int:
        """
        Select a single action based on the given Q-values.

        Parameters
        ----------
        q_values : np.ndarray
            Array of Q-values for each action.

        Returns
        -------
        int
            The selected action.
        """
        pass

    @abstractmethod
    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
        """
        Select actions for a batch of Q-value arrays.

        Parameters
        ----------
        q_values : np.ndarray
            2D array of Q-values, where each row corresponds to a set of Q-values for one instance.

        Returns
        -------
        np.ndarray
            Array of selected actions for each instance in the batch.
        """
        pass

    @abstractmethod
    def update_params(self) -> None:
        """
        Update the parameters of the exploration policy.

        This method is typically used to adjust exploration parameters over time.
        """
        pass


class EpsilonGreedyPolicy(ExplorationPolicy):
    """
    Epsilon-Greedy exploration policy.

    Selects actions randomly with probability epsilon, otherwise selects the action with the highest Q-value.
    """

    def __init__(
        self,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9999,
        decay_type: Literal["exponential", "linear", "fixed"] = "exponential",
    ) -> None:
        """
        Initialize an EpsilonGreedyPolicy instance.

        Parameters
        ----------
        epsilon : float, default=1.0
            The initial exploration rate.
        epsilon_min : float, default=0.01
            The minimum exploration rate.
        epsilon_decay : float, default=0.9999
            The decay rate for epsilon.
        decay_type : {'exponential', 'linear', 'fixed'}, default='exponential'
            The type of decay for epsilon.
        """
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_type = decay_type

    def select_action(self, q_values: np.ndarray) -> int:
        """
        Select a single action using the epsilon-greedy strategy.

        Parameters
        ----------
        q_values : np.ndarray
            Array of Q-values for each action.

        Returns
        -------
        int
            The selected action.
        """
        num_actions = q_values.size
        if np.random.rand() <= self.epsilon:
            return np.random.choice(num_actions)
        action = np.argmax(q_values)
        return action

    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
        """
        Select actions for a batch of Q-value arrays using the epsilon-greedy strategy.

        Parameters
        ----------
        q_values : np.ndarray
            2D array of Q-values, where each row corresponds to a set of Q-values for one instance.

        Returns
        -------
        np.ndarray
            Array of selected actions for each instance in the batch.
        """
        batch_size = q_values.shape[0]
        num_actions = q_values.shape[1]
        # Exploration: random actions
        random_actions = np.random.choice(num_actions, batch_size)
        # Exploitation: predicted actions
        greedy_actions = np.argmax(q_values, axis=1)
        # Epsilon-greedy policy
        mask = np.random.rand(batch_size) <= self.epsilon
        actions = np.where(mask, random_actions, greedy_actions)
        return actions

    def update_params(self) -> None:
        """
        Update the epsilon parameter based on the decay type.

        Adjusts epsilon according to the specified decay type ('exponential', 'linear', or 'fixed').
        """
        if self.decay_type == "exponential":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif self.decay_type == "linear":
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        elif self.decay_type == "fixed":
            return  # No update for fixed epsilon
        else:
            raise ValueError(
                f"Not valid decay type '{self.decay_type}'. Valid types are 'exponential', 'linear' or 'fixed'."
            )


class BoltzmannPolicy(ExplorationPolicy):
    """
    Boltzmann exploration policy.

    Selects actions probabilistically based on a softmax distribution of Q-values.
    """

    def __init__(self, tau=1.0) -> None:
        """
        Initialize a BoltzmannPolicy instance.

        Parameters
        ----------
        tau : float, default=1.0
            The temperature parameter controlling exploration.
        """
        self.tau = tau

    def select_action(self, q_values: np.ndarray) -> int:
        """
        Select a single action using the Boltzmann strategy.

        Parameters
        ----------
        q_values : np.ndarray
            Array of Q-values for each action.

        Returns
        -------
        int
            The selected action.
        """
        num_actions = q_values.size
        exp_values = np.exp(q_values / self.tau)
        probabilities = exp_values / np.sum(exp_values)
        action = np.random.choice(num_actions, p=probabilities)
        return action

    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
        """
        Select actions for a batch of Q-value arrays using the Boltzmann strategy.

        Parameters
        ----------
        q_values : np.ndarray
            2D array of Q-values, where each row corresponds to a set of Q-values for one instance.

        Returns
        -------
        np.ndarray
            Array of selected actions for each instance in the batch.
        """
        batch_size = q_values.shape[0]
        num_actions = q_values.shape[1]
        exp_values = np.exp(q_values / self.tau)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        actions = np.array(
            [
                np.random.choice(num_actions, p=probabilities[i])
                for i in range(batch_size)
            ]
        )
        return actions

    def update_params(self) -> None:
        """
        Update the parameters of the Boltzmann policy.

        This method is a no-op for BoltzmannPolicy as it does not require parameter updates.
        """
        return
