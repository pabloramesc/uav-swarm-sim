"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from abc import ABC, abstractmethod
from typing import Literal
import numpy as np


class ExplorationPolicy(ABC):
    @abstractmethod
    def select_action(self, q_values: np.ndarray) -> int:
        pass

    @abstractmethod
    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_params(self) -> None:
        pass


class EpsilonGreedyPolicy(ExplorationPolicy):
    def __init__(
        self,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9999,
        decay_type: Literal["exponential", "linear", "fixed"] = "exponential",
    ) -> None:
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_type = decay_type

    def select_action(self, q_values: np.ndarray) -> int:
        num_actions = q_values.size
        if np.random.rand() <= self.epsilon:
            return np.random.choice(num_actions)
        action = np.argmax(q_values)
        return action

    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
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
    def __init__(self, tau=1.0) -> None:
        self.tau = tau

    def select_action(self, q_values: np.ndarray) -> int:
        num_actions = q_values.size
        exp_values = np.exp(q_values / self.tau)
        probabilities = exp_values / np.sum(exp_values)
        action = np.random.choice(num_actions, p=probabilities)
        return action

    def select_action_batch(self, q_values: np.ndarray) -> np.ndarray:
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
        return
