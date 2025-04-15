"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    """
    Represents a single experience tuple in reinforcement learning.

    Attributes
    ----------
    state : np.ndarray
        The state observed.
    action : int
        The action taken.
    next_state : np.ndarray
        The state observed after taking the action.
    reward : float
        The reward received after taking the action.
    done : bool
        Whether the episode ended after taking the action.
    """

    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool

    def to_tuple(self) -> tuple[np.ndarray, int, np.ndarray, float, bool]:
        """
        Convert the experience to a tuple.

        Returns
        -------
        tuple
            A tuple representation of the experience.
        """
        return (self.state, self.action, self.next_state, self.reward, self.done)


@dataclass
class ExperiencesBatch:
    """
    Represents a batch of experiences for training.

    Attributes
    ----------
    states : np.ndarray
        Array of states.
    next_states : np.ndarray
        Array of next states.
    actions : np.ndarray
        Array of actions taken.
    rewards : np.ndarray
        Array of rewards received.
    dones : np.ndarray
        Array of done flags indicating episode termination.
    indices : np.ndarray, optional
        Array of indices of the experiences in the buffer.
    weights : np.ndarray, optional
        Array of importance sampling weights.
    """

    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    indices: np.ndarray = None
    weights: np.ndarray = None

    @property
    def size(self) -> int:
        """
        The number of experiences in the batch.
        """
        return self._check_consistency()

    def to_experiences(self) -> list[Experience]:
        """
        Convert the batch to a list of Experience objects.

        Returns
        -------
        list[Experience]
            A list of Experience objects.
        """
        size = self._check_consistency()
        experiences = [None] * size
        for i in range(size):
            experiences[i] = Experience(
                state=self.states[i],
                next_state=self.next_states[i],
                action=self.actions[i],
                reward=self.rewards[i],
                done=self.dones[i],
            )
        return experiences

    def _check_consistency(self) -> int:
        """
        Check the consistency of the batch attributes.
        """
        if self.states.shape != self.next_states.shape:
            raise ValueError("States and next states shapes must be equal")

        size = self.states.shape[0]

        if self.actions.ndim > 1 or self.actions.shape[0] != size:
            raise ValueError(f"Actions must be a 1D array of size {size}")

        if self.rewards.ndim > 1 or self.rewards.shape[0] != size:
            raise ValueError(f"Rewards must be a 1D array of size {size}")

        if self.dones.ndim > 1 or self.dones.shape[0] != size:
            raise ValueError(f"Dones must be a 1D array of size {size}")

        if self.indices is None:
            return size

        if self.indices.ndim > 1 or self.indices.shape[0] != size:
            raise ValueError(f"Indices must be a 1D array of size {size}")

        if self.weights is None:
            return size

        if self.weights.ndim > 1 or self.weights.shape[0] != size:
            raise ValueError(f"Weights must be a 1D array of size {size}")

        return size
