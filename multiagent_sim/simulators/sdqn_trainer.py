import logging
import random
import numpy as np

from multiagent_sim.mobility.sdqn_position_controller import SDQNConfig
from multiagent_sim.sdqn.frames.frame_generator import FrameGeneratorFactory

from ..mobility.utils import environment_random_positions
from ..sdqn import SDQNBrain, SDQNWrapper
from .sdqn_simulator import SDQNSimulator

logger = logging.getLogger(__name__)


class SDQNTrainer(SDQNSimulator):

    def __init__(
        self,
        num_drones: int,
        num_users: int = 0,
        sdqn_config: SDQNConfig | None = None,
        frame_factory: FrameGeneratorFactory | None = None,
        model_path: str | None = None,
    ) -> None:
        super().__init__(num_drones, num_users, sdqn_config, frame_factory, model_path)
        self.cum_rewards = np.zeros(num_drones)

    def _create_sdqn_brain(self) -> SDQNBrain:
        frame_shape = self.frame_factory.create(env=None).shape
        wrapper = SDQNWrapper(
            frame_shape=frame_shape,
            model_path=self.model_path,
            train_mode=True,
        )
        brain = SDQNBrain(wrapper=wrapper, environment=self.environment)
        return brain

    def initialize(self) -> None:
        super().initialize()
        self.cum_rewards = np.zeros(self.sim.drones.size)

    def update(self, dt: float | None = None) -> None:
        self.update_drone_positions()

        self.sim.update(dt)

        self.sdqn_brain.train_step(
            drone_positions=self.sim.drone_states[:, 0:3],
            user_positions=self.sim.user_states[:, 0:3],
        )

        self.reset_collided_drones()

        if self.sdqn_brain.rewards is None:
            raise RuntimeError("SDQN Brain not initialized.")
        self.cum_rewards += self.sdqn_brain.rewards

    def reset_collided_drones(self) -> None:
        dones = self.sdqn_brain.dones
        if dones is None:
            return

        done_indices = np.arange(self.sim.drones.size)[dones]
        for i in done_indices:
            state = np.zeros(6)
            state[0:3] = environment_random_positions(
                num_positions=1, env=self.environment
            )
            drone = self.sim.drones[i]
            drone.initialize(state)

            logger.warning(f"⚠️  Reset drone {i} to initial states")

    @property
    def training_status_str(self) -> str:
        return self.sdqn_brain.wrapper.training_status_str()

    def train(
        self,
        max_episodes: int = 1000,
        max_episode_steps: int | None = None,
        max_episode_time: float | None = None,
        num_obstacles: int = 0,
    ) -> None:
        for episode in range(1, max_episodes + 1):
            self.create_random_environment(num_obstacles)
            self.initialize()

            step, terminated = 0, False
            while not terminated:
                step += 1
                self.update()

                if max_episode_steps is not None and step >= max_episode_steps:
                    terminated = True

                if (
                    max_episode_time is not None
                    and self.sim.clock.sim_time >= max_episode_time
                ):
                    terminated = True

    def create_random_environment(self, num_obstacles: int = 0) -> None:
        self.environment.clear_obstacles()
        for _ in range(num_obstacles):
            if np.random.rand() > 0.5:
                self._add_circular_obstacle()
            else:
                self._add_rectangular_obstacle()

    def _add_circular_obstacle(self) -> None:
        center = np.random.uniform(
            self.environment.boundary.bounds.xy_min,
            self.environment.boundary.bounds.xy_max,
            size=(2,),
        )
        min_size = min(self.environment.boundary.bounds.size)
        radius = np.random.uniform(0.02 * min_size, 0.2 * min_size)
        self.environment.add_circular_obstacle(center, radius)

    def _add_rectangular_obstacle(self) -> None:
        bottom_left = np.random.uniform(
            self.environment.boundary.bounds.xy_min,
            self.environment.boundary.bounds.xy_max,
            size=(2,),
        )
        min_size = min(self.environment.boundary.bounds.size)
        width_height = np.random.uniform(0.05 * min_size, 0.5 * min_size, size=(2,))
        top_right = bottom_left + width_height
        self.environment.add_rectangular_obstacle(bottom_left, top_right)
