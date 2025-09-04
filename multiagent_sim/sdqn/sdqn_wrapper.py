"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import os
from datetime import datetime
import time

import numpy as np
from dqn import DQNAgentPER, EpsilonGreedyPolicy, ExperiencesBatch
from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    InputLayer,
    Rescaling,
    Input,
    MaxPooling2D,
)
from keras.losses import Huber
from keras.models import Model, Sequential, save_model, load_model
from keras.optimizers import Adam


class SDQNWrapper:

    def __init__(
        self,
        frame_shape: tuple[int, int, int],
        num_actions: int = 5,
        model_path: str = None,
        train_mode: bool = True,
    ):
        self.frame_shape = frame_shape
        self.num_actions = num_actions
        self.model_path = model_path
        self.train_mode = train_mode

        # if len(self.frame_shape) == 2:
        #     self.frame_shape += (1,)

        if len(self.frame_shape) != 3:
            raise ValueError("Frame shape must be (height, width, channels)")

        if not self.train_mode and self.model_path is None:
            raise ValueError("Model path shall be provided if not in training mode")

        # If not model path provided, create new model file with timestamp
        if self.model_path is None:
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            self.model_path = f"sdqn-model-{timestamp}.keras"

        # If model doesn't exist, create new model and save it
        if not os.path.exists(self.model_path):
            model = self.build_keras_model()
            save_model(model=model, filepath=self.model_path)

        # Create linear decaying epsilon-greedy policy with initial random exploration
        if self.train_mode:
            policy = EpsilonGreedyPolicy(
                epsilon=1.0, epsilon_min=0.1, epsilon_decay=1e-5, decay_type="linear"
            )
        # If not in training mode, use fixed policy with no exploration (epsilon=0)
        else:
            policy = EpsilonGreedyPolicy(
                epsilon=0.0, epsilon_min=0.0, decay_type="fixed"
            )

        # Create DQN Agent and set the model
        model = load_model(filepath=self.model_path, compile=True)
        self.dqn_agent = DQNAgentPER(
            model=model,
            batch_size=64,
            gamma=0.99,
            policy=policy,
            memory_size=500_000,
            update_freq=5000,
        )
        self.dqn_agent.model.summary()

        if self.frame_shape != self.dqn_agent.model.input_shape[1:]:
            raise ValueError("Frame shape does not match model input shape")

        if self.num_actions != self.dqn_agent.model.output_shape[1]:
            raise ValueError("The number of actions does not match the output size")

        self.min_train_samples = 50_000
        self.autosave_freq = 1000

        self.train_metrics: dict = None
        self.train_t0 = None

    def add_experiences(
        self,
        frames: np.ndarray,
        actions: np.ndarray,
        next_frames: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a batch of experiences to the agent's memory.

        Args:
            frames: Array of frames.
            actions: Array of actions taken.
            next_frames: Array of next frames.
            rewards: Array of rewards received.
            dones: Array of done flags indicating episode termination.

        Raises:
            ValueError: If the frames or next frames are not valid.
            Warning: If called while not in training mode.
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
        """Train the agent using the experiences in memory and returns metrics
        dictionary with training performance indicators. Autosave model if needed.

        Returns:
            Training metrics, or None if training is not performed.
        """
        if not self.train_mode:
            return None

        if self.dqn_agent.memory.size < self.min_train_samples:
            return None

        self.train_metrics = self.dqn_agent.train()

        if self.train_t0 is None:
            self.train_t0 = time.time()

        if self.dqn_agent.train_steps % self.autosave_freq:
            self.dqn_agent.model.save(filepath=self.model_path, overwrite=True)

        return self.train_metrics

    def act(self, frames: np.ndarray) -> np.ndarray:
        self.check_frames(frames)
        actions = self.dqn_agent.act_on_batch(frames)
        return actions

    def build_keras_model(self) -> Model:
        inputs = Input(shape=self.frame_shape, dtype="uint8")
        x = Rescaling(1.0 / 255.0)(inputs)

        # Block 1
        x = Conv2D(32, 3, strides=1, padding="same", activation="relu")(x)
        x = Conv2D(32, 3, strides=1, padding="same", activation="relu")(x)
        x = MaxPooling2D(2)(x)
        # x = Dropout(0.1)(x)

        # Block 2
        x = Conv2D(64, 3, strides=1, padding="same", activation="relu")(x)
        x = Conv2D(64, 3, strides=1, padding="same", activation="relu")(x)
        x = MaxPooling2D(2)(x)
        # x = Dropout(0.1)(x)

        # Block 3
        x = Conv2D(128, 3, strides=1, padding="same", activation="relu")(x)
        x = Conv2D(128, 3, strides=1, padding="same", activation="relu")(x)
        x = MaxPooling2D(2)(x)
        # x = Dropout(0.1)(x)

        # Dense head
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        # x = Dropout(0.2)(x)
        outputs = Dense(self.num_actions, activation="linear")(x)

        model = Model(inputs=inputs, outputs=outputs, name="DQN_model")

        model.compile(
            optimizer=Adam(learning_rate=0.00025),
            loss=Huber(delta=1.0),
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
        if self.train_t0 is None:
            return np.nan
        time.time() - self.train_t0

    @property
    def train_speed(self) -> float:
        return self.train_steps / self.train_elapsed

    @property
    def memory_size(self) -> int:
        return self.dqn_agent.memory.size

    @property
    def epsilon(self) -> float:
        return self.dqn_agent.policy.epsilon

    @property
    def loss(self) -> float:
        if self.train_metrics is None:
            return np.nan
        return self.train_metrics.get("loss", np.nan)

    def training_status_str(self) -> str:
        return (
            f"Train steps: {self.train_steps}, "
            f"Train time: {self.train_elapsed:.0f} s, "
            f"Train speed: {self.train_speed:.2f} sps, "
            f"Memory size: {self.memory_size}, "
            f"Epsilon: {self.epsilon:.4f}, "
            f"Loss: {self.loss:.4e}"
        )
