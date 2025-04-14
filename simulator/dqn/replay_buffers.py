import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np
from collections import deque


@dataclass
class Experience:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool

    def to_tuple(self) -> tuple[np.ndarray, int, np.ndarray, float, bool]:
        return (self.state, self.action, self.next_state, self.reward, self.done)


@dataclass
class ExperiencesBatch:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    indices: np.ndarray = None
    weights: np.ndarray = None

    @property
    def size(self) -> int:
        return self._check_consistency()

    def to_experiences(self) -> list[Experience]:
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


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        self.max_size = int(max_size)
        self.buffer: deque[Experience] = deque(maxlen=self.max_size)

    def add(self, exp: Experience) -> None:
        self.buffer.append(exp)

    def add_batch(self, batch: list[Experience]) -> None:
        self.buffer.extend(batch)

    def sample(self, batch_size: int) -> ExperiencesBatch:
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = self._get_batch_by_indices(indices)
        return batch

    def _get_batch_by_indices(self, indices: list[int]) -> ExperiencesBatch:
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
        return len(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        max_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        min_priority: float = 1e-6,
    ):
        super().__init__(max_size)

        self.priorities: deque[float] = deque(maxlen=self.max_size)

        self.alpha = alpha
        self.beta = beta
        self.min_priority = min_priority

    def add(self, exp: Experience, td_error: float = 1.0) -> None:
        priority = max(self.min_priority, td_error)
        self.priorities.append(priority)
        super().add(exp)

    def sample(self, batch_size: int) -> ExperiencesBatch:
        priorities = np.array(self.priorities) ** self.alpha + self.min_priority
        probabilities = priorities / np.sum(priorities)

        indices = np.random.choice(self.size, size=batch_size, p=probabilities)
        batch = self._get_batch_by_indices(indices)

        weights = (self.size * probabilities[indices]) ** -self.beta
        batch.weights = weights / weights.max()

        return batch

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
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
        dones=np.random.choice([True, False], (num_exps,))
    )
    
    memory.add_batch(batch.to_experiences())
    
    indices = np.random.choice(memory.size, size=64, replace=False)
    
    pass