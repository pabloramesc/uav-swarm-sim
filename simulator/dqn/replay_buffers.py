"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

from collections import deque

import numpy as np

from .experiences import Experience, ExperiencesBatch


class ReplayBuffer:
    """
    ReplayBuffer class for storing and sampling experiences.

    Stores experiences in a fixed-size buffer and allows sampling for training.
    """

    def __init__(self, max_size: int) -> None:
        """
        Initialize a ReplayBuffer instance.

        Parameters
        ----------
        max_size : int
            The maximum number of experiences the buffer can hold.
        """
        self.max_size = int(max_size)
        self.buffer: deque[Experience] = deque(maxlen=self.max_size)

    def add(self, exp: Experience) -> None:
        """
        Add a single experience to the buffer.

        Parameters
        ----------
        exp : Experience
            The experience to add to the buffer.
        """
        self.buffer.append(exp)

    def add_batch(self, batch: list[Experience]) -> None:
        """
        Add a batch of experiences to the buffer.

        Parameters
        ----------
        batch : list[Experience]
            A list of experiences to add to the buffer.
        """
        self.buffer.extend(batch)

    def sample(self, batch_size: int) -> ExperiencesBatch:
        """
        Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of experiences to sample.

        Returns
        -------
        ExperiencesBatch
            A batch of sampled experiences.
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = self._get_batch_by_indices(indices)
        return batch

    def _get_batch_by_indices(self, indices: list[int]) -> ExperiencesBatch:
        """
        Retrieve a batch of experiences by their indices.
        """
        experiences = [self.buffer[i].to_tuple() for i in indices]
        states, actions, next_states, rewards, dones = zip(*experiences)
        batch = ExperiencesBatch(
            states=np.array(states),
            actions=np.array(actions),
            next_states=np.array(next_states),
            rewards=np.array(rewards),
            dones=np.array(dones),
            indices=indices,
            weights=None,
        )
        return batch

    @property
    def size(self) -> int:
        """
        The number of experiences in the buffer.
        """
        return len(self.buffer)

    def __len__(self) -> int:
        """
        The number of experiences in the buffer.
        """
        return len(self.buffer)


class PriorityReplayBuffer(ReplayBuffer):
    """
    PriorityReplayBuffer class for storing and sampling experiences with priorities.

    Extends ReplayBuffer to include prioritized sampling based on TD errors.
    """

    def __init__(
        self,
        max_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        min_priority: float = 1e-6,
    ):
        """
        Initialize a PriorityReplayBuffer instance.

        Parameters
        ----------
        max_size : int
            The maximum number of experiences the buffer can hold.
        alpha : float, default=0.6
            The exponent used to scale priorities for sampling.
        beta : float, default=0.4
            The exponent used to scale importance sampling weights.
        min_priority : float, default=1e-6
            The minimum priority value to avoid zero probabilities.
        """
        super().__init__(max_size)

        self.priorities: deque[float] = deque(maxlen=self.max_size)
        self.alpha = alpha
        self.beta = beta
        self.min_priority = min_priority

    def add(self, exp: Experience, td_error: float = 1.0) -> None:
        """
        Add a single experience with a priority to the buffer.

        Parameters
        ----------
        exp : Experience
            The experience to add to the buffer.
        td_error : float, default=1.0
            The TD error used to calculate the priority.
        """
        priority = max(self.min_priority, td_error)
        self.priorities.append(priority)
        super().add(exp)

    def sample(self, batch_size: int) -> ExperiencesBatch:
        """
        Sample a batch of experiences from the buffer based on their priorities.

        Parameters
        ----------
        batch_size : int
            The number of experiences to sample.

        Returns
        -------
        ExperiencesBatch
            A batch of sampled experiences.
        """
        priorities = np.array(self.priorities) ** self.alpha + self.min_priority
        probabilities = priorities / np.sum(priorities)

        indices = np.random.choice(self.size, size=batch_size, p=probabilities)
        batch = self._get_batch_by_indices(indices)

        weights = (self.size * probabilities[indices]) ** -self.beta
        batch.weights = weights / weights.max()

        return batch

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        """
        Update the priorities of experiences in the buffer.

        Parameters
        ----------
        indices : list[int]
            The indices of the experiences to update.
        td_errors : np.ndarray
            The new TD errors for the experiences.
        """
        for i in indices:
            self.priorities[i] = max(self.min_priority, td_errors[i])


if __name__ == "__main__":
    num_exps = 1_000_000

    memory = ReplayBuffer(max_size=num_exps)

    batch = ExperiencesBatch(
        states=np.random.rand(*(num_exps, 10)),
        next_states=np.random.rand(*(num_exps, 10)),
        actions=np.random.randint(0, 10, (num_exps,)),
        rewards=np.random.rand(*(num_exps,)),
        dones=np.random.choice([True, False], (num_exps,)),
    )

    memory.add_batch(batch.to_experiences())

    indices = np.random.choice(memory.size, size=64, replace=False)

    pass
