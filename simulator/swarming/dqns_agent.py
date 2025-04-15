"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
import keras.api as kr

from dqn import DQNAgent, EpsilonGreedyPolicy, ExperiencesBatch


class DQNSAgent:
    """
    Deep Q-Learning Swarming Agent (DQNSAgent).

    Manages the training and inference of a DQN-based agent for controlling a swarm of drones.
    """

    def __init__(self, num_drones: int, train: bool = False, model_path: str = None):
        """
        Initialize the DQNSAgent.

        Parameters
        ----------
        num_drones : int
            The number of drones in the swarm.
        train : bool, default=False
            Whether the agent is in training mode.
        model_path : str, optional
            Path to the pre-trained model. Required if `train` is False.

        Raises
        ------
        ValueError
            If `num_drones` is less than or equal to 0.
        Exception
            If `train` is False and `model_path` is not provided.
        """
        if num_drones <= 0:
            raise ValueError("The number of drones must be greater than 0")
        self.num_drones = num_drones

        self.num_cells = 100
        self.num_actions = 9
        self.state_shape = (self.num_cells, self.num_cells, 2)
        self.states_shape = (self.num_drones, self.num_cells, self.num_cells, 2)

        if not train and model_path is None:
            raise Exception("Model path must be provided in no training mode.")

        if train and model_path is None:
            model = self.build_keras_model()
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            self.model_path = f"dqns-model-{timestamp}.keras"
            kr.models.save_model(model, self.model_path)

        policy = EpsilonGreedyPolicy(
            epsilon=1.0 if train else 0.0,
            epsilon_min=0.1 if train else 0.0,
        )
        self.dqn_agent = DQNAgent(
            model=None,
            batch_size=256,
            gamma=0.95,
            policy=policy,
            memory_size=100_000,
            update_steps=10_000,
            autosave_steps=1000,
            file_name=self.model_path,
            verbose=train,
        )
        self.dqn_agent.load_model()

    def build_keras_model(self) -> kr.Model:
        """
        Build a Keras model for the DQNS (Deep Q-Learning Swarming).

        The model processes the state input and outputs Q-values for each action.

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

    def act(self, states: np.ndarray) -> None:
        """
        Perform actions based on the current states.

        Parameters
        ----------
        states : np.ndarray
            The current states of the drones with shape (num_drones, num_cells, num_cells, 2).

        Raises
        ------
        ValueError
            If the states are not valid.
        """
        self._check_states(states)
        self.dqn_agent.act_batch(states)
        
    def add_experiences(self, batch: ExperiencesBatch) -> None:
        """
        Add a batch of experiences to the agent's memory.

        Parameters
        ----------
        batch : ExperiencesBatch
            A batch of experiences containing states, actions, rewards, next states, and done flags.

        Raises
        ------
        ValueError
            If the states or next states in the batch are not valid.
        """
        self._check_states(batch.states)
        self._check_states(batch.next_states)
        self.dqn_agent.update_memory_batch(batch)

    def train(self) -> dict:
        """
        Train the agent using the experiences in memory.

        Returns
        -------
        dict
            Training metrics, or None if training is not performed.
        """
        if not self.train:
            return

        if self.dqn_agent.memory.size < self.min_train_samples:
            return

        metrics = self.dqn_agent.train()
        return metrics

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
            If the states are not of dtype uint8 or do not match the expected shape.
        """
        if not states.dtype(np.uint8):
            raise ValueError("States must be an uint8 numpy array.")
        if states.shape != self.states_shape:
            raise ValueError(f"States shape must be {self.states_shape}")
