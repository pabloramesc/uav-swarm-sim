"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import os
from datetime import datetime

import keras.api as kr
import numpy as np
import tensorflow as tf
from dqn import DQNAgent, EpsilonGreedyPolicy, ExperiencesBatch, PriorityReplayBuffer


class SDQNWrapper:

    def __init__(
        self,
        frame_shape: tuple[int, int, int],
        num_actions: int,
        model_path: str = None,
        train_mode: bool = True,
    ):
        self.frame_shape = frame_shape
        self.num_actions = num_actions
        self.model_path = model_path
        self.train_mode = train_mode

        if not train_mode and self.model_path is None:
            raise ValueError("Model path shall be provided if not in training mode")

        if self.model_path is None:
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            self.model_path = f"sdqn-model-{timestamp}.keras"

        if not os.path.exists(self.model_path):
            self.model = self.build_keras_model()
            kr.models.save_model(self.model, self.model_path)

        if train_mode:
            self.policy = EpsilonGreedyPolicy(
                epsilon=1.0, epsilon_min=0.1, epsilon_decay=1e-5, decay_type="linear"
            )
        else:
            self.policy = EpsilonGreedyPolicy(
                epsilon=0.0, epsilon_min=0.0, decay_type="fixed"
            )

        self.memory = PriorityReplayBuffer(max_size=100_000, beta_annealing=0.0)

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
        )
        self.dqn_agent.load_model(self.model_path, compile=True)
        self.dqn_agent.model.summary()

        if self.frame_shape != self.dqn_agent.model.input_shape[1:]:
            raise ValueError("Frame shape does not match model input shape")

        if self.num_actions != self.dqn_agent.model.output_shape[1]:
            raise ValueError("The number of actions does not match the output size")

        self.train_metrics: dict = None
        self.min_train_samples = 10_000

    def add_experiences(
        self,
        frames: np.ndarray,
        actions: np.ndarray,
        next_frames: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        Add a batch of experiences to the agent's memory.

        Parameters
        ----------
        frames : np.ndarray
            Array of frames.
        actions : np.ndarray
            Array of actions taken.
        next_frames : np.ndarray
            Array of next frames.
        rewards : np.ndarray
            Array of rewards received.
        dones : np.ndarray
            Array of done flags indicating episode termination.

        Raises
        ------
        ValueError
            If the frames or next frames are not valid.
        """
        if not self.train_mode:
            raise Warning("Do not add experiences in no training mode!")

        self.check_frames(frames)
        self.check_frames(next_frames)
        batch = ExperiencesBatch(
            states=frames,
            next_states=next_frames,
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
        if not self.train_mode:
            return None

        if self.dqn_agent.memory.size < self.min_train_samples:
            return None

        self.train_metrics = self.dqn_agent.train()
        return self.train_metrics

    def act(self, frames: np.ndarray) -> np.ndarray:
        self.check_frames(frames)
        actions = self.dqn_agent.act_on_batch(frames)
        return actions

    def build_keras_model(self) -> kr.Model:
        model = kr.models.Sequential(
            [
                kr.layers.InputLayer(shape=self.frame_shape, dtype="uint8"),
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

    def check_frame(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            raise ValueError("Frame must be an uint8 numpy array.")
        if frame.shape != self.frame_shape:
            raise ValueError(f"Frame shape must be {self.frame_shape}")

    def check_frames(self, frames: np.ndarray) -> None:
        self.check_frame(frames[0])

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
        if self.train_metrics is not None and "accuracy" in self.train_metrics:
            return self.train_metrics["accuracy"]
        return np.nan

    @property
    def loss(self) -> float:
        if self.train_metrics is not None and "loss" in self.train_metrics:
            return self.train_metrics["loss"]
        return np.nan

    def training_status_str(self) -> str:
        return (
            f"Train steps: {self.train_steps}, "
            f"Train speed: {self.train_speed:.2f} sps, "
            f"Memory size: {self.memory_size}, "
            f"Epsilon: {self.epsilon:.4f}, "
            f"Loss: {self.loss:.4e}, "
            f"Accuracy: {self.accuracy*100:.2f} %"
        )
