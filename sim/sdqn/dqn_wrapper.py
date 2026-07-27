"""Thin, optional integration with the external ``dqn-lab`` package.

The simulator and frame/reward modules intentionally do not import this module's
machine-learning dependencies.  Even importing this module is lightweight:
TensorFlow, Keras, and dqn-lab are loaded only when a :class:`DQNWrapper` is
constructed.
"""

from __future__ import annotations

import builtins
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .actions import NUM_ACTIONS

DecayType = Literal["fixed", "linear", "exponential"]


@dataclass(frozen=True)
class DQNConfig:
    memory_size: int = 200_000
    min_memory: int = 50_000
    update_freq: int = 10_000
    batch_size: int = 32
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 1e-5
    decay_type: DecayType = "linear"
    n_step: int = 3
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_annealing: float = 1e-6
    autosave_freq: int = 1_000
    learning_rate: float = 2.5e-4

    def __post_init__(self) -> None:
        for name in (
            "memory_size",
            "min_memory",
            "update_freq",
            "batch_size",
            "n_step",
            "autosave_freq",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if self.memory_size <= 0:
            raise ValueError("memory_size must be positive.")
        if not 0 <= self.min_memory <= self.memory_size:
            raise ValueError("min_memory must be between 0 and memory_size.")
        if self.update_freq <= 0 or self.batch_size <= 0 or self.n_step <= 0:
            raise ValueError("update_freq, batch_size, and n_step must be positive.")
        if self.autosave_freq < 0:
            raise ValueError("autosave_freq cannot be negative.")
        if self.decay_type not in ("fixed", "linear", "exponential"):
            raise ValueError(f"Unknown epsilon decay type: {self.decay_type!r}.")

        for name in ("gamma", "epsilon", "epsilon_min", "per_alpha", "per_beta"):
            value = _finite_real(getattr(self, name), name=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.epsilon_min > self.epsilon:
            raise ValueError("epsilon_min cannot exceed epsilon.")
        if _finite_real(self.epsilon_decay, name="epsilon_decay") < 0.0:
            raise ValueError("epsilon_decay cannot be negative.")
        if _finite_real(self.per_beta_annealing, name="per_beta_annealing") < 0.0:
            raise ValueError("per_beta_annealing cannot be negative.")
        if _finite_real(self.learning_rate, name="learning_rate") <= 0.0:
            raise ValueError("learning_rate must be positive.")


def _load_ml_dependencies() -> SimpleNamespace:
    """Import dqn-lab and Keras, working around dqn-lab's bare ``@profile``.

    Some dqn-lab modules use the line-profiler decorator without importing it.
    Python's line-profiler normally injects ``profile`` into ``builtins``.  For
    normal execution we temporarily provide the equivalent no-op decorator
    before importing the package, without modifying the vendored submodule.
    """

    sentinel = object()
    previous_profile = getattr(builtins, "profile", sentinel)
    if previous_profile is sentinel:
        builtins.profile = lambda function: function

    try:
        from dqn import EpsilonGreedyPolicy, ExperiencesBatch, RainbowDQN
        from dqn.layers import DuelingHead
        from keras.layers import Conv2D, Dense, Flatten, Input, Rescaling
        from keras.losses import Huber
        from keras.models import Model, load_model, save_model
        from keras.optimizers import Adam
    except ImportError as exc:
        raise ImportError(
            "SDQN model support requires dqn-lab, Keras, and TensorFlow. "
            "The simulation environment itself can be used without them."
        ) from exc
    finally:
        if previous_profile is sentinel:
            del builtins.profile
        else:
            builtins.profile = previous_profile

    return SimpleNamespace(
        EpsilonGreedyPolicy=EpsilonGreedyPolicy,
        ExperiencesBatch=ExperiencesBatch,
        RainbowDQN=RainbowDQN,
        DuelingHead=DuelingHead,
        Conv2D=Conv2D,
        Dense=Dense,
        Flatten=Flatten,
        Input=Input,
        Rescaling=Rescaling,
        Huber=Huber,
        Model=Model,
        load_model=load_model,
        save_model=save_model,
        Adam=Adam,
    )


class DQNWrapper:
    """Small adapter between batched SDQN frames and dqn-lab."""

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        frame_shape: tuple[int, int, int],
        num_actions: int = NUM_ACTIONS,
        model_path: str | Path | None = None,
        train_mode: bool = False,
        config: DQNConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.frame_shape = tuple(int(value) for value in frame_shape)
        self.num_actions = int(num_actions)
        self.train_mode = bool(train_mode)
        self.config = config or DQNConfig(**kwargs)

        if len(self.frame_shape) != 3 or any(value <= 0 for value in self.frame_shape):
            raise ValueError("Frame shape must contain three positive dimensions.")
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive.")

        if model_path is None:
            if not self.train_mode:
                raise ValueError("model_path is required in inference mode.")
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            self.model_path = Path("outputs/models") / f"sdqn-{timestamp}.keras"
            self.logger.info("Using new model path '%s'.", self.model_path)
        else:
            self.model_path = Path(model_path)

        if not self.train_mode and not self.model_path.is_file():
            raise FileNotFoundError(f"SDQN model does not exist: {self.model_path}")

        self._ml = _load_ml_dependencies()
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model_path.is_file():
            model = self._ml.load_model(
                filepath=self.model_path,
                custom_objects={"DuelingHead": self._ml.DuelingHead},
                compile=True,
            )
            self.logger.info("Loaded model from '%s'.", self.model_path)
        else:
            model = self.build_keras_model()
            self._ml.save_model(model=model, filepath=self.model_path)
            self.logger.info("Saved new model to '%s'.", self.model_path)

        if model.optimizer is None:
            model.compile(
                optimizer=self._ml.Adam(learning_rate=self.config.learning_rate),
                loss=self._ml.Huber(delta=1.0),
            )
        else:
            model.optimizer.learning_rate.assign(self.config.learning_rate)

        if self.train_mode:
            self.policy = self._ml.EpsilonGreedyPolicy(
                epsilon=self.config.epsilon,
                epsilon_min=self.config.epsilon_min,
                epsilon_decay=self.config.epsilon_decay,
                decay_type=self.config.decay_type,
            )
        else:
            self.policy = self._ml.EpsilonGreedyPolicy(
                epsilon=0.0, epsilon_min=0.0, decay_type="fixed"
            )

        self.dqn_agent = self._ml.RainbowDQN(
            model=model,
            policy=self.policy,
            batch_size=self.config.batch_size,
            memory_size=self.config.memory_size,
            update_freq=self.config.update_freq,
            gamma=self.config.gamma,
            n_step=self.config.n_step,
            alpha=self.config.per_alpha,
            beta=self.config.per_beta,
            beta_annealing=self.config.per_beta_annealing,
        )

        input_shape = tuple(self.dqn_agent.model.input_shape[1:])
        if self.frame_shape != input_shape:
            raise ValueError(
                f"Frame shape {self.frame_shape} does not match model input "
                f"shape {input_shape}."
            )
        output_size = int(self.dqn_agent.model.output_shape[-1])
        if self.num_actions != output_size:
            raise ValueError(
                f"num_actions ({self.num_actions}) does not match model output "
                f"size {output_size}."
            )

        self.train_t0: float | None = None
        self.train_metrics: dict[str, float] = {}

    def act(self, frames: np.ndarray) -> NDArray[np.int32]:
        """Select one discrete action for each frame."""

        self.check_frames(frames)
        actions = self.dqn_agent.act_on_batch(frames, training=self.train_mode)
        return np.asarray(actions, dtype=np.int32)

    def add_experiences(
        self,
        frames: NDArray,
        actions: NDArray[np.int32],
        next_frames: NDArray,
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
        truncated: NDArray[np.bool_] | None = None,
    ) -> None:
        """Add a vectorized transition to replay memory."""

        if not self.train_mode:
            raise RuntimeError("Cannot add experiences in inference mode.")
        self.check_frames(frames)
        self.check_frames(next_frames)
        batch = self._ml.ExperiencesBatch(
            states=frames,
            next_states=next_frames,
            actions=actions,
            rewards=rewards,
            dones=dones,
            truncated=truncated,
        )
        self.dqn_agent.add_experiences_batch(batch)

    def train(self) -> dict[str, float]:
        """Run one optimizer update when replay memory is warm enough."""

        if not self.train_mode:
            raise RuntimeError("Cannot train a model configured for inference.")
        if self.memory_size < self.config.min_memory:
            return {}

        metrics = self.dqn_agent.train()
        self.train_metrics = metrics if metrics is not None else {}
        if self.train_t0 is None:
            self.train_t0 = time.monotonic()

        autosave = self.config.autosave_freq
        if autosave > 0 and self.train_steps > 0 and self.train_steps % autosave == 0:
            self.save()
        return self.train_metrics

    def save(self) -> None:
        """Save the current online network."""

        self._ml.save_model(model=self.dqn_agent.model, filepath=self.model_path)
        self.logger.info("Saved model to '%s'.", self.model_path)

    def flush_experiences(self) -> None:
        """Flush pending n-step transitions at an episode boundary."""

        flush = getattr(self.dqn_agent.memory, "flush", None)
        if flush is not None:
            flush()

    def build_keras_model(self) -> Any:
        """Build the default convolutional dueling network."""

        inputs = self._ml.Input(shape=self.frame_shape, dtype="uint8")
        features = self._ml.Rescaling(1.0 / 255.0)(inputs)
        features = self._ml.Conv2D(32, 8, strides=4, activation="relu")(features)
        features = self._ml.Conv2D(64, 4, strides=2, activation="relu")(features)
        features = self._ml.Conv2D(64, 3, strides=1, activation="relu")(features)
        features = self._ml.Flatten()(features)

        value = self._ml.Dense(512, activation="relu")(features)
        value = self._ml.Dense(1, activation="linear")(value)
        advantage = self._ml.Dense(512, activation="relu")(features)
        advantage = self._ml.Dense(self.num_actions, activation="linear")(advantage)
        output = self._ml.DuelingHead(dtype="float32")([value, advantage])

        model = self._ml.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=self._ml.Adam(learning_rate=self.config.learning_rate),
            loss=self._ml.Huber(delta=1.0),
        )
        return model

    def check_frame(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            raise ValueError("Frame must have dtype uint8.")
        if frame.shape != self.frame_shape:
            raise ValueError(f"Frame shape must be {self.frame_shape}.")

    def check_frames(self, frames: np.ndarray) -> None:
        if frames.dtype != np.uint8:
            raise ValueError("Frames must have dtype uint8.")
        if frames.ndim != 4 or tuple(frames.shape[1:]) != self.frame_shape:
            raise ValueError(f"Frames must have shape (N, {self.frame_shape!r}).")

    @property
    def train_steps(self) -> int:
        return int(self.dqn_agent.train_steps)

    @property
    def train_elapsed(self) -> float:
        return 0.0 if self.train_t0 is None else time.monotonic() - self.train_t0

    @property
    def train_speed(self) -> float:
        elapsed = self.train_elapsed
        return 0.0 if elapsed <= 0.0 else self.train_steps / elapsed

    @property
    def memory_size(self) -> int:
        return int(self.dqn_agent.memory.size)

    @property
    def epsilon(self) -> float:
        return float(self.policy.epsilon)

    @property
    def loss(self) -> float:
        return float(self.train_metrics.get("loss", float("nan")))

    @property
    def training_status_str(self) -> str:
        elapsed = _format_time(self.train_elapsed)
        return (
            f"Train steps: {self.train_steps}, "
            f"Train time: {elapsed}, "
            f"Train speed: {self.train_speed:.2f} sps, "
            f"Memory size: {self.memory_size}, "
            f"Epsilon: {self.epsilon:.4f}, "
            f"Loss: {self.loss:.4e}"
        )


def _format_time(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _finite_real(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number
