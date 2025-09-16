import logging
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
from dqn import DQNAgentPER, EpsilonGreedyPolicy, ExperiencesBatch
from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    InputLayer,
    MaxPooling2D,
    Rescaling,
)
from keras.losses import Huber
from keras.models import Model, Sequential, load_model, save_model
from keras.optimizers import Adam


class SDQNWrapper:

    logger = logging.getLogger("SDQNWrapper")
    logger.setLevel(logging.INFO)

    def __init__(
        self,
        frame_shape: tuple[int, int, int],
        num_actions: int = 5,
        train_mode: bool = True,
        model_path: Optional[str] = None,
        min_memory_size: int = 10_000,
        autosave_freq: int = 1000,
    ):
        self.frame_shape = frame_shape
        self.num_actions = int(num_actions)
        self.train_mode = bool(train_mode)
        self.model_path = model_path
        self.min_memory_size = int(min_memory_size)
        self.autosave_freq = int(autosave_freq)

        if len(self.frame_shape) != 3:
            raise ValueError("Frame shape must be (height, width, channels).")

        if not self.train_mode and self.model_path is None:
            raise ValueError("Model path shall be provided if not in training mode.")

        # If not model path provided, create new model file with timestamp
        if self.model_path is None:
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            self.model_path = f"sdqn-model-{timestamp}.keras"
            self.logger.info(f"New model path created: {self.model_path}")

        # If model exists, load it
        if os.path.exists(self.model_path):
            model = load_model(filepath=self.model_path, compile=True)
            self.logger.info(f"Model loaded from '{self.model_path}'.")
        # If not, build model and save it
        else:
            model = self.build_keras_model()
            save_model(model=model, filepath=self.model_path)
            self.logger.info(f"Model saved to '{self.model_path}'.")

        # Create linear decaying epsilon-greedy policy with initial random exploration
        if self.train_mode:
            self.policy = EpsilonGreedyPolicy(
                epsilon=1.0, epsilon_min=0.01, epsilon_decay=1e-5, decay_type="linear"
            )
        # If not in training mode, use fixed policy with no exploration (epsilon=0)
        else:
            self.policy = EpsilonGreedyPolicy(
                epsilon=0.0, epsilon_min=0.0, decay_type="fixed"
            )

        # Create DQN Agent and set the model
        self.dqn_agent = DQNAgentPER(
            model=model,  # type: ignore
            batch_size=32,
            gamma=0.99,
            policy=self.policy,
            memory_size=100_000,
            update_freq=5000,
        )
        self.dqn_agent.model.summary()

        # Check model input and output shapes
        input_shape = self.dqn_agent.model.input_shape[1:]
        if self.frame_shape != input_shape:
            raise ValueError(
                f"Frame shape {self.frame_shape} does not match model input shape {input_shape}."
            )

        output_size = self.dqn_agent.model.output_shape[1]
        if self.num_actions != output_size:
            raise ValueError(
                f"The number of actions ({self.num_actions}) does not match the output size {output_size}."
            )

        self.train_t0 = None
        self.train_metrics: dict[str, float] = dict()

    def act(self, frames: np.ndarray) -> np.ndarray:
        self.check_frames(frames)
        actions = self.dqn_agent.act_on_batch(frames)
        return actions

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

    def train(self) -> dict[str, float]:
        """Train the agent using the experiences in memory and returns metrics
        dictionary with training performance indicators. Autosave model if needed.

        Returns:
            Dictionary with training metrics, empty if training is not performed.
        """
        if not self.train_mode:
            raise RuntimeError(
                "Cannot call train method. Model not configured for training."
            )

        if self.dqn_agent.memory.size < self.min_memory_size:
            return {}

        metrics = self.dqn_agent.train()
        self.train_metrics = metrics if metrics is not None else {}

        if self.train_t0 is None:
            self.train_t0 = time.time()

        if (
            self.dqn_agent.train_steps > 0
            and self.dqn_agent.train_steps % self.autosave_freq == 0
        ):
            save_model(model=self.dqn_agent.model, filepath=self.model_path)
            self.logger.info(f"Model saved to '{self.model_path}'.")

        return self.train_metrics

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
            optimizer=Adam(learning_rate=0.00025),  # type: ignore
            loss=Huber(delta=1.0),
        )

        return model

    def check_frame(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            raise ValueError("Frame must be an uint8 numpy array.")
        if frame.shape != self.frame_shape:
            raise ValueError(f"Frame shape must be {self.frame_shape}")

    def check_frames(self, frames: np.ndarray) -> None:
        if frames.dtype != np.uint8:
            raise ValueError("Frame must be an uint8 numpy array.")
        expected_shape = (None, *self.frame_shape)
        if frames.shape[1:] != expected_shape[1:]:
            raise ValueError(f"Frame shape must be {expected_shape}")

    @property
    def train_steps(self) -> int:
        return self.dqn_agent.train_steps

    @property
    def train_elapsed(self) -> float:
        if self.train_t0 is None:
            return 0.0
        return time.time() - self.train_t0

    @property
    def train_speed(self) -> float:
        if self.train_elapsed <= 0:
            return 0.0
        return self.train_steps / self.train_elapsed

    @property
    def memory_size(self) -> int:
        return self.dqn_agent.memory.size

    @property
    def epsilon(self) -> float:
        return self.policy.epsilon

    @property
    def loss(self) -> float:
        loss = self.train_metrics.get("loss", float("nan"))
        return loss

    @property
    def training_status_str(self) -> str:
        return (
            f"Train steps: {self.train_steps}, "
            f"Train time: {self.train_elapsed:.0f} s, "
            f"Train speed: {self.train_speed:.2f} sps, "
            f"Memory size: {self.memory_size}, "
            f"Epsilon: {self.epsilon:.4f}, "
            f"Loss: {self.loss:.4e}"
        )
