"""Training orchestration for the maintained SDQN environment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..utils.csv_logger import CSVLogger
from .actions import NUM_ACTIONS
from .dqn_wrapper import DQNConfig, DQNWrapper
from .environment import SDQNEnvironment, SDQNEnvironmentConfig


class TrainableBatchPolicy(Protocol):
    def act(self, frames: np.ndarray) -> NDArray[np.integer]: ...

    def add_experiences(
        self,
        frames: NDArray,
        actions: NDArray[np.int32],
        next_frames: NDArray,
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
        truncated: NDArray[np.bool_] | None = None,
    ) -> None: ...

    def train(self) -> dict[str, float]: ...


class SDQNTrainer:
    """Coordinate experience collection without owning simulation transitions."""

    _LOG_COLUMNS = [
        "episode",
        "sim_time",
        "sim_steps",
        "users_coverage",
        "mean_reward",
        "memory_size",
        "epsilon",
        "loss",
        "train_time",
        "train_steps",
        "train_speed",
    ]

    def __init__(
        self,
        environment_config: SDQNEnvironmentConfig | None = None,
        dqn_config: DQNConfig | None = None,
        *,
        environment: SDQNEnvironment | None = None,
        policy: TrainableBatchPolicy | None = None,
        model_path: str | Path | None = None,
        log_path: str | Path | None = None,
        render: bool = False,
    ) -> None:
        if environment is not None and environment_config is not None:
            raise ValueError("Pass either environment or environment_config, not both.")

        self.environment = environment or SDQNEnvironment(environment_config)
        self.dqn_config = dqn_config or DQNConfig()
        self.dqn: TrainableBatchPolicy = policy or DQNWrapper(
            frame_shape=self.environment.frame_shape,
            num_actions=NUM_ACTIONS,
            model_path=model_path,
            train_mode=True,
            config=self.dqn_config,
        )

        self.csv_logger = (
            CSVLogger(
                filepath=str(log_path),
                columns=self._LOG_COLUMNS,
                header_lines=[str(self.environment.config), str(self.dqn_config)],
                if_exists="version",
            )
            if log_path is not None
            else None
        )

        self.gui: Any | None = None
        if render:
            from ..gui.sdqn_viewer import SDQNViewer

            self.gui = SDQNViewer(self.environment, fps=10.0, background_type="rssi")

        self.frames: NDArray[np.uint8] | None = None
        self.cumulative_rewards = np.zeros(
            self.environment.num_drones, dtype=np.float64
        )
        self._coverage_sum = 0.0
        self._episode_steps = 0
        self._total_steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> NDArray[np.uint8]:
        self.frames, _ = self.environment.reset(seed=seed, options=options)
        self.cumulative_rewards.fill(0.0)
        self._coverage_sum = 0.0
        self._episode_steps = 0
        if self.gui is not None:
            self.gui.reset()
        return self.frames

    def step(
        self,
        *,
        max_steps: int | None = None,
        max_time: float | None = None,
    ) -> bool:
        """Collect one transition and report whether the episode limit was hit."""

        if self.frames is None:
            raise RuntimeError("Call reset() before step().")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when provided.")
        if max_time is not None and max_time <= 0.0:
            raise ValueError("max_time must be positive when provided.")

        actions = np.asarray(self.dqn.act(self.frames), dtype=np.int32)
        next_frames, rewards, terminated, truncated, _ = self.environment.step(actions)

        reached_step_limit = (
            max_steps is not None and self.environment.step_count >= max_steps
        )
        reached_time_limit = max_time is not None and self.environment.time >= max_time
        reached_limit = reached_step_limit or reached_time_limit
        if reached_limit:
            truncated = np.ones_like(terminated, dtype=np.bool_)
        episode_finished = bool(terminated.any() or truncated.any())

        self.dqn.add_experiences(
            self.frames,
            actions,
            next_frames,
            rewards,
            terminated,
            truncated,
        )
        self.frames = next_frames
        self.cumulative_rewards += rewards
        self._episode_steps += 1
        self._total_steps += 1
        if self.environment.metrics is not None:
            self._coverage_sum += self.environment.metrics.users_coverage
        return episode_finished

    def train_step(self) -> dict[str, float]:
        return self.dqn.train()

    def render(self, *, force: bool = False) -> None:
        if self.gui is None:
            raise RuntimeError("Rendering was not configured.")
        self.gui.render(force=force)

    def train(
        self,
        *,
        train_freq: int = 1,
        max_episodes: int = 1_000,
        max_episode_steps: int | None = 1_000,
        max_episode_time: float | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Run a finite training job and return one summary per episode."""

        if train_freq <= 0:
            raise ValueError("train_freq must be positive.")
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive.")
        if max_episode_steps is None and max_episode_time is None:
            raise ValueError(
                "At least one episode step/time limit is required for train()."
            )

        history: list[dict[str, float]] = []
        for episode in range(1, max_episodes + 1):
            episode_seed = None if seed is None else seed + episode - 1
            self.reset(seed=episode_seed)

            finished = False
            while not finished:
                finished = self.step(
                    max_steps=max_episode_steps,
                    max_time=max_episode_time,
                )
                if self._total_steps % train_freq == 0:
                    self.train_step()
                if self.gui is not None:
                    self.render()
                if verbose and (finished or self._episode_steps % 10 == 0):
                    ending = "\n" if finished else "\r"
                    print(
                        f"Episode: {episode}, {self.training_status_str}",
                        end=ending,
                    )

            flush = getattr(self.dqn, "flush_experiences", None)
            if flush is not None:
                flush()

            summary = self._episode_summary(episode)
            history.append(summary)
            if self.csv_logger is not None:
                self.csv_logger.log(**summary)

        save = getattr(self.dqn, "save", None)
        if save is not None:
            save()
        return history

    def close(self) -> None:
        if self.gui is not None:
            self.gui.close()
        self.environment.close()

    @property
    def avg_users_coverage(self) -> float:
        if self._episode_steps == 0:
            return 0.0
        return self._coverage_sum / self._episode_steps

    @property
    def training_status_str(self) -> str:
        model_status = getattr(self.dqn, "training_status_str", "")
        prefix = (
            f"Sim steps: {self.environment.step_count}, "
            f"Sim time: {self.environment.time:.2f} s, "
            f"Avg coverage: {self.avg_users_coverage * 100.0:.2f} %, "
            f"Mean reward: {self.cumulative_rewards.mean():.2f}"
        )
        return f"{prefix}, {model_status}" if model_status else prefix

    def _episode_summary(self, episode: int) -> dict[str, float]:
        return {
            "episode": float(episode),
            "sim_time": self.environment.time,
            "sim_steps": float(self.environment.step_count),
            "users_coverage": self.avg_users_coverage,
            "mean_reward": float(self.cumulative_rewards.mean()),
            "memory_size": float(getattr(self.dqn, "memory_size", 0)),
            "epsilon": float(getattr(self.dqn, "epsilon", float("nan"))),
            "loss": float(getattr(self.dqn, "loss", float("nan"))),
            "train_time": float(getattr(self.dqn, "train_elapsed", 0.0)),
            "train_steps": float(getattr(self.dqn, "train_steps", 0)),
            "train_speed": float(getattr(self.dqn, "train_speed", 0.0)),
        }
