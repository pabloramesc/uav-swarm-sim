"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import os
from datetime import datetime

import numpy as np
import keras.api as kr

from dqn import DQNAgent, EpsilonGreedyPolicy, ExperiencesBatch, PriorityReplayBuffer


class CentralAgent:
    """
    Deep Q-Learning Swarming Agent (DQNSAgent).

    Manages the training and inference of a DQN-based agent for controlling a swarm of drones.
    """

    def __init__(
        self,
        num_drones: int,
        num_cells: int = 100,
        num_actions: int = 9,
        num_channels: int = 1,
        training_mode: bool = False,
        model_path: str = None,
    ):
        """
        Initialize the DQNSAgent.

        Parameters
        ----------
        num_drones : int
            The number of drones in the swarm.
        training_mode : bool, default=False
            Whether the agent is in training mode.
        model_path : str, optional
            Path to the pre-trained model. Required if `train` is False.

        Raises
        ------
        Exception
            If `train` is False and `model_path` is not provided.
        """
        self.num_drones = num_drones
        self.num_cells = num_cells
        self.num_actions = num_actions
        self.num_channels = num_channels

        self.state_shape = (self.num_cells, self.num_cells, self.num_channels)
        self.states_shape = (self.num_drones, *self.state_shape)

        if not training_mode and model_path is None:
            raise Exception("Model path must be provided in no training mode.")
        self.training_mode = training_mode
        self.model_path = model_path

        if self.model_path is None:
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            self.model_path = f"dqns-model-{timestamp}.keras"

        if not os.path.exists(self.model_path):
            model = self.build_keras_model()
            kr.models.save_model(model, self.model_path)

        self.policy = EpsilonGreedyPolicy(
            epsilon=1.0 if self.training_mode else 0.0,
            epsilon_min=0.1 if self.training_mode else 0.0,
            epsilon_decay=1e-5 if self.training_mode else 0.0,
            decay_type="linear" if self.training_mode else "fixed",
        )
        self.memory = PriorityReplayBuffer(max_size=500_000, beta_annealing=0.0)
        self.dqn_agent = DQNAgent(
            model=None,
            batch_size=64,
            gamma=0.99,
            policy=self.policy,
            # memory_size=500_000,
            memory=self.memory,
            update_steps=5_000,
            autosave_steps=1000,
            file_name=self.model_path,
            verbose=self.training_mode,
        )
        self.dqn_agent.load_model(self.model_path, compile=True)
        self.dqn_agent.model.summary()

        self.train_metrics: dict = None
        self.min_train_samples = 10_000

    @property
    def drone_positions(self) -> np.ndarray:
        return self.drone_states[:, 0:2]

    @property
    def train_steps(self) -> int:
        return self.dqn_agent.train_steps

    @property
    def train_elapsed(self) -> float:
        return self.dqn_agent.train_elapsed

    @property
    def train_speed(self) -> float:
        return self.dqn_agent.train_speed or np.nan

    @property
    def memory_size(self) -> int:
        return self.dqn_agent.memory.size

    @property
    def epsilon(self) -> float:
        return self.policy.epsilon

    @property
    def accuracy(self) -> float:
        if self.train_metrics and "accuracy" in self.train_metrics:
            return self.train_metrics["accuracy"]
        return np.nan

    @property
    def loss(self) -> float:
        if self.train_metrics and "loss" in self.train_metrics:
            return self.train_metrics["loss"]
        return np.nan

    def build_keras_model(self) -> kr.Model:
        """
        Build a Keras model for the DQNS (Deep Q-Learning Swarming).

        The model processes the state input and outputs Q-values for each
        action.

        Returns
        -------
        keras.Model
            The compiled Keras model.
        """

        model = kr.models.Sequential(
            [
                kr.layers.InputLayer(shape=self.state_shape, dtype="uint8"),
                kr.layers.Rescaling(1.0 / 255.0),
                kr.layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu"),
                kr.layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu"),
                kr.layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu"),
                kr.layers.Flatten(),
                kr.layers.Dense(512, activation="relu"),
                kr.layers.Dense(self.num_actions, activation="linear"),
            ]
        )

        model.compile(
            optimizer=kr.optimizers.Adam(learning_rate=0.00025),
            loss=kr.losses.Huber(delta=1.0),
            metrics=["accuracy"],
        )

        return model

    def act(self, states: np.ndarray) -> np.ndarray:
        """
        Perform actions based on the current states.

        Parameters
        ----------
        states : np.ndarray
            The current states of the drones with shape (num_drones, num_cells,
            num_cells, 2).

        Raises
        ------
        ValueError
            If the states are not valid.
        """
        self._check_states(states)
        actions = self.dqn_agent.act_on_batch(states)
        return actions

    def add_experiences(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        Add a batch of experiences to the agent's memory.

        Parameters
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

        Raises
        ------
        ValueError
            If the states or next states are not valid.
        """
        self._check_states(states)
        self._check_states(next_states)
        batch = ExperiencesBatch(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )
        self.dqn_agent.add_experiences_batch(batch)

    def train(self) -> dict:
        """
        Train the agent using the experiences in memory.

        It also updates and returns metrics dictionary with training
        performance indicators.

        Returns
        -------
        dict
            Training metrics, or None if training is not performed.
        """
        if not self.training_mode:
            return

        if self.dqn_agent.memory.size < self.min_train_samples:
            return

        self.train_metrics = self.dqn_agent.train()
        return self.train_metrics

    def _check_states(self, states: np.ndarray) -> None:
        """
        Validate the states array.

        Parameters
        ----------
        states : np.ndarray
            The states to validate.

        Raises
        ------
        ValueError
            If the states are not of dtype uint8 or do not match the expected
            shape.
        """
        if not states.dtype == np.uint8:
            raise ValueError("States must be an uint8 numpy array.")
        if states.shape != self.states_shape:
            raise ValueError(f"States shape must be {self.states_shape}")
