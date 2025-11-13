import numpy as np

from typing import Optional

from sim.sdqn import DQNWrapper, DQNConfig
from .sdqn_gym_env import GymEnvironment, GymConfig
from ..gui.sdqn_viewer import SDQNViewer
from sim.utils import CSVLogger


class SDQNTrainer:

    def __init__(
        self,
        gym_config: Optional[GymConfig] = None,
        dqn_config: Optional[DQNConfig] = None,
        model_path: Optional[str] = None,
        log_path: Optional[str] = None,
        render: bool = False,
    ) -> None:
        self.gym_config = gym_config or GymConfig()
        self.dqn_config = dqn_config or DQNConfig()

        self.env = GymEnvironment(config=self.gym_config)
        self.dqn = DQNWrapper(
            frame_shape=self.env.frame_factory.shape,
            num_actions=5,
            model_path=model_path,
            train_mode=True,
            config=self.dqn_config,
        )

        if log_path is not None:
            self.csv_logger = CSVLogger(
                filepath=log_path,
                columns=[
                    "episode",
                    "sim_time",
                    "sim_steps",
                    "user_coverage",
                    "mean_reward",
                    "memory_size",
                    "epsilon",
                    "loss",
                    "train_time",
                    "train_steps",
                    "train_speed",
                ],
                header_lines=[str(self.gym_config), str(self.dqn_config)],
                if_exists="overwrite",
            )
        else:
            self.csv_logger = None

        self.cum_rewards = np.zeros(self.gym_config.num_drones)

        if render:
            self.gui = SDQNViewer(sdqn=self.env, fps=1.0, background_type="rssi")
        else:
            self.gui = None

    def reset(self) -> None:
        self.frames = self.env.reset()
        self._cum_users_coverage = 0.0
        self._episode_steps = 0
        if self.gui is not None:
            self.gui.reset()

    def step(
        self, max_steps: Optional[int] = None, max_time: Optional[float] = None
    ) -> bool:
        actions = self.dqn.act(self.frames)

        next_frames, rewards, dones = self.env.step(actions)

        terminated = False

        if max_steps and self.env.sim_step >= max_steps:
            terminated = True

        elif max_time and self.env.sim_time >= max_time:
            terminated = True

        if terminated:
            truncs = np.ones_like(rewards, dtype=bool)
        else:
            truncs = np.zeros_like(rewards, dtype=bool)

        self.dqn.add_experiences(
            self.frames, actions, next_frames, rewards, dones, truncs
        )

        self.frames = next_frames
        self.cum_rewards += rewards
        self.cum_rewards[dones] = 0.0

        self._cum_users_coverage += self.env.metrics.users_coverage
        self._episode_steps += 1

        return terminated

    def train_step(self) -> dict:
        return self.dqn.train()

    def render(self):
        if self.gui is None:
            raise RuntimeError("GUI not configured.")
        self.gui.render()

    def train(
        self,
        train_freq: int = 1,
        max_episodes: int = 1000,
        max_episode_steps: int | None = None,
        max_episode_time: float | None = None,
        verbose: bool = True,
        render: bool = False,
    ) -> None:
        total_steps = 0
        for episode in range(1, max_episodes + 1):
            self.reset()

            step, terminated = 0, False
            while not terminated:
                step += 1
                total_steps += 1

                terminated = self.step(
                    max_steps=max_episode_steps, max_time=max_episode_steps
                )

                if total_steps % train_freq == 0:
                    self.train_step()

                # If episode is terminated flush all n-step buffers
                if terminated:
                    self.dqn.dqn_agent.memory.flush()

                # Print each 10 steps or if terminated
                if verbose and (terminated or step % 10 == 0):
                    print(f"Episode: {episode}, " + self.training_status_str, end="\r")

                if render and self.gui is not None:
                    self.gui.render()

            print()

            if self.csv_logger is not None:
                self.csv_logger.log(
                    episode=episode,
                    sim_time=self.env.sim_time,
                    sim_steps=self.env.sim_step,
                    user_coverage=self.avg_users_coverage,
                    mean_reward=self.cum_rewards.mean(),
                    epsilon=self.dqn.epsilon,
                    loss=self.dqn.loss,
                    memory_size=self.dqn.memory_size,
                    train_time=self.dqn.train_elapsed,
                    train_steps=self.dqn.train_steps,
                    train_speed=self.dqn.train_speed,
                )

            self.cum_rewards = np.zeros(self.gym_config.num_drones)

    @property
    def avg_users_coverage(self) -> float:
        if self._episode_steps > 0:
            return self._cum_users_coverage / self._episode_steps
        else:
            return 0.0

    @property
    def training_status_str(self) -> str:
        return (
            f"Sim steps: {self.env.sim_step}, "
            f"Sim time: {self.env.sim_time:.2f} s, "
            f"Coverage: {self.env.metrics.users_coverage*100:.2f} %, "
            f"Avg Cov: {self.avg_users_coverage*100:.2f} %, "
            f"Mean reward: {self.cum_rewards.mean():.2f}, "
            + self.dqn.training_status_str
        )
