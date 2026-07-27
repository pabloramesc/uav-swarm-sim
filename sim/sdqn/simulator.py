"""Policy-driven SDQN inference simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..environment import Environment
from .actions import NUM_ACTIONS
from .dqn_wrapper import DQNWrapper
from .environment import SDQNEnvironment, SDQNEnvironmentConfig
from .frames import FrameGeneratorFactory


class BatchPolicy(Protocol):
    def act(self, frames: np.ndarray) -> NDArray[np.integer]: ...


class SDQNSimulator(SDQNEnvironment):
    """Run inference through the exact same transition path used for training."""

    def __init__(
        self,
        config: SDQNEnvironmentConfig | None = None,
        *,
        model_path: str | Path | None = None,
        policy: BatchPolicy | None = None,
        environment: Environment | None = None,
        frame_factory: FrameGeneratorFactory | None = None,
    ) -> None:
        super().__init__(
            config=config,
            environment=environment,
            frame_factory=frame_factory,
        )
        self.policy: BatchPolicy = policy or DQNWrapper(
            frame_shape=self.frame_shape,
            num_actions=NUM_ACTIONS,
            model_path=model_path,
            train_mode=False,
        )
        self.last_rewards = np.zeros(self.num_drones, dtype=np.float32)
        self.last_terminated = np.zeros(self.num_drones, dtype=np.bool_)
        self.last_truncated = np.zeros(self.num_drones, dtype=np.bool_)

    def step(
        self,
        actions: NDArray[np.integer] | list[int] | tuple[int, ...] | None = None,
    ) -> tuple[
        NDArray[np.uint8],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict[str, Any],
    ]:
        if actions is None:
            if self.last_frames is None:
                raise RuntimeError("Call reset() before policy-driven step().")
            actions = self.policy.act(self.last_frames)

        transition = super().step(actions)
        (
            _,
            self.last_rewards,
            self.last_terminated,
            self.last_truncated,
            _,
        ) = transition
        return transition

    @property
    def simulation_status_str(self) -> str:
        if self.metrics is None:
            return "Simulation not initialized"
        return (
            f"Sim time: {self.time:.2f} s, "
            f"Sim steps: {self.step_count}, "
            f"Area coverage: {self.metrics.area_coverage * 100.0:.2f} %, "
            f"Users coverage: {self.metrics.users_coverage * 100.0:.2f} %, "
            f"Directly connected: "
            f"{self.metrics.direct_connections * 100.0:.2f} %, "
            f"Globally connected: "
            f"{self.metrics.global_connections * 100.0:.2f} %"
        )
