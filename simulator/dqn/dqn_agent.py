"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import keras.api as kr
import numpy as np

from ..dqn.exploration_policies import EpsilonGreedyPolicy, ExplorationPolicy
from ..dqn.replay_buffers import Experience, ExperiencesBatch, PriorityReplayBuffer, ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        model: kr.Model = None,
        batch_size: int = 256,
        gamma: float = 0.95,
        policy: ExplorationPolicy = None,
        memory: ReplayBuffer = None,
        memory_size: int = 10_000,
        update_steps: int = 1000,
        autosave_steps: int = 0,
        file_name: str = None,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.gamma = gamma

        if self.model is not None:
            self.set_model(model)

        self.memory: ReplayBuffer = (
            memory if memory is not None else ReplayBuffer(memory_size)
        )

        if self.memory.max_size < batch_size:
            raise ValueError(
                f"Memory max size {self.memory.max_size} cannot be smaller than batch size {self.batch_size}"
            )

        self.policy: ExplorationPolicy = (
            policy if policy is not None else EpsilonGreedyPolicy()
        )

        self.update_steps = update_steps
        self.autosave_steps = autosave_steps
        self.file_name = file_name
        self.verbose = verbose

        self.train_steps = int(0)

    def act(self, state: np.ndarray) -> int:
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)
        action = self.policy.select_action(q_values[0])
        return action

    def act_batch(self, states: np.ndarray) -> np.ndarray:
        if states.ndim == 3:  # (84, 84, 4) -> (1, 84, 84, 4)
            states = np.expand_dims(states, axis=0)
        q_values = self.model.predict(states, verbose=0)
        actions = self.policy.select_action_batch(q_values)
        return actions

    def update_memory(self, exp: Experience) -> None:
        self.memory.add(exp)

    def update_memory_batch(self, batch: ExperiencesBatch) -> None:
        experiences = batch.to_experiences()
        self.memory.add_batch(experiences)

    def train(self) -> dict:
        if self.memory.size < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        q_values = self.model.predict(batch.states, verbose=0)
        q_values_next = self.target_model.predict(batch.next_states, verbose=0)

        done_mask = np.array(batch.dones, dtype=bool)
        q_target = np.where(
            done_mask,
            batch.rewards,
            batch.rewards + self.gamma * np.max(q_values_next, axis=1),
        )

        if isinstance(self.memory, PriorityReplayBuffer):
            q_actual = q_values[np.arange(self.batch_size), batch.actions]
            td_errors = q_target - q_actual
            self.memory.update_priorities(batch.indices, td_errors)

        q_values[np.arange(self.batch_size), batch.actions] = q_target
        metrics = self.model.train_on_batch(
            batch.states, q_values, sample_weight=batch.weights, return_dict=True
        )

        self.policy.update_params()

        self.train_steps += 1

        if self.update_steps > 0 and self.train_steps % self.update_steps == 0:
            self.update_target_model()

        if self.autosave_steps > 0 and self.train_steps % self.autosave_steps == 0:
            self.save_model()

        return metrics

    def set_model(self, model: kr.Model) -> None:
        self.model = model
        self.target_model = kr.models.clone_model(model)
        self.target_model.set_weights(model.get_weights())

    def update_target_model(self) -> None:
        self.target_model.set_weights(self.model.get_weights())
        if self.verbose:
            print("DQN Agent: Target model updated.")

    def save_model(self, file_name: str = None) -> None:
        file_name = file_name or self.file_name
        self.model.save(file_name)
        if self.verbose:
            print(f"DQN Agent: Model saved to '{file_name}'.")

    def load_model(self, file_name: str = None, compile: bool = True) -> None:
        file_name = file_name or self.file_name
        model = kr.models.load_model(file_name, compile=compile)
        self.set_model(model)
        if self.verbose:
            print(f"DQN Agent: Model loaded from '{file_name}'.")
