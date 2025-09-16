import numpy as np

from sim.sdqn import SDQNWrapper
from .sdqn_gym_env import SDQNGymEnv
from ..gui.sdqn_viewer import SDQNViewer


class SDQNTrainer:

    def __init__(
        self,
        num_drones: int = 16,
        num_users: int = 32,
        model_path: str | None = None,
        render: bool = False,
    ) -> None:
        self.num_drones = int(num_drones)
        self.num_users = int(num_users)

        self.env = SDQNGymEnv(dt=1.0, drones_speed=10.0)
        self.wrapper = SDQNWrapper(
            frame_shape=self.env.frame_factory.shape,
            num_actions=5,
            model_path=model_path,
            train_mode=True,
        )

        self.cum_rewards = np.zeros(self.num_drones)

        if render:
            self.gui = SDQNViewer(sdqn=self.env, fps=1.0, background_type="rssi")
        else:
            self.gui = None

    def reset(self) -> None:
        self.frames = self.env.reset(
            num_drones=self.num_drones, num_users=self.num_users
        )
        if self.gui is not None:
            self.gui.initialize()

    def step(self) -> bool:
        actions = self.wrapper.act(self.frames)

        next_frames, rewards, dones = self.env.step(actions)

        self.wrapper.add_experiences(self.frames, actions, next_frames, rewards, dones)

        self.frames = next_frames
        self.cum_rewards += rewards

        terminated = any(dones)
        return terminated

    def train_step(self) -> dict:
        return self.wrapper.train()

    def render(self):
        if self.gui is None:
            raise RuntimeError("GUI not configured.")
        self.gui.update()

    def train(
        self,
        train_freq: int = 1,
        max_episodes: int = 1000,
        max_episode_steps: int | None = None,
        max_episode_time: float | None = None,
        verbose: bool = True,
        render: bool = False,
    ) -> None:
        for episode in range(1, max_episodes + 1):
            self.reset()

            step, terminated = 0, False
            while not terminated:
                step += 1

                terminated = self.step()

                if step % train_freq == 0:
                    self.train_step()

                if max_episode_steps is not None and step >= max_episode_steps:
                    terminated = True

                if (
                    max_episode_time is not None
                    and self.env.sim_time >= max_episode_time
                ):
                    terminated = True

                if verbose:
                    print(f"Episode: {episode}, " + self.training_status_str, end="\r")

                if render and self.gui is not None:
                    self.gui.update()

            print()

    @property
    def training_status_str(self) -> str:
        return (
            f"Sim steps: {self.env.sim_step}, "
            f"Sim time: {self.env.sim_time:.2f} s, "
            f"Area cov: {self.env.metrics.area_coverage*100:.2f} %, "
            f"Users cov: {self.env.metrics.users_coverage*100:.2f} %, "
            f"Mean reward: {self.cum_rewards.mean():.2f}, "
            + self.wrapper.training_status_str
        )
