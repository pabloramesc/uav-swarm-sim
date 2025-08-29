from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..agents import Drone
from ..mobility.sdqn_position_controller import SDQNConfig
from ..mobility.swarm_position_controller import DummyPositionController
from ..sdqn import (
    RewardManager,
    SDQNBrain,
    SDQNWrapper,
    SDQNInterface,
)
from ..sdqn.frames import FrameBase, FrameFactory
from .multiagent_simulator import MultiAgentSimulator
from ..mobility.utils import environment_random_positions, grid_positions


class SDQNTrainer(MultiAgentSimulator):

    def __init__(
        self,
        num_drones: int,
        num_users: int = 0,
        dt: float = 0.01,
        sdqn_config: SDQNConfig = None,
        frame_factory: FrameFactory = None,
        model_path: str = None,
    ) -> None:
        self.sdqn_config = sdqn_config or SDQNConfig()
        self.frame_factory = frame_factory or FrameFactory()
        self.model_path = model_path

        self.sdqn_brain = self._create_sdqn_brain()

        super().__init__(
            num_drones=num_drones,
            num_users=num_users,
            num_gcs=1,
            dt=dt,
            dem_path=None,
            use_network=False,
        )

        self.reward_manager = RewardManager(env=self.environment)

        self.prev_frames: np.ndarray = None
        self.prev_actions: np.ndarray = None

        self.displacement = sdqn_config.target_velocity * dt

    def _create_sdqn_brain(self) -> SDQNBrain:
        frame_shape = self.frame_factory.create().shape
        wrapper = SDQNWrapper(
            frame_shape=frame_shape,
            model_path=self.model_path,
            train_mode=True,
        )
        return SDQNBrain(wrapper)

    def _create_sdqn_interface(self, iface_id: int) -> SDQNInterface:
        frame_generator = self.frame_factory.create()
        interface = SDQNInterface(iface_id, frame_generator)
        self.sdqn_brain.register_interface(interface)
        return interface

    def _create_drone(self, sdqn_config: SDQNConfig = None, **kwargs) -> Drone:
        iface = self._create_sdqn_interface(iface_id=len(self.agents))
        dummy_controller = DummyPositionController(
            config=self.sdqn_config, env=self.environment
        )
        drone = Drone(
            agent_id=len(self.agents),
            environment=self.environment,
            position_controller=dummy_controller,
            drones_registry=self.drones,
            users_registry=self.users,
            neighbor_provider="registry",
        )
        return drone

    def initialize(self, home: ArrayLike = [0.0, 0.0], spacing: float = 5.0) -> None:
        self.logger.info("Initializing simulation ...")

        # Initilaize GSC (Ground Control Station)
        gcs_states = np.zeros((1,6))
        gcs_states[0, 0:2] = np.asarray(home[0:2])
        gcs_states[0, 2] = self.environment.get_elevation(home[0:2])
        self.gcs.initialize(states=gcs_states)

        # Initialize drones
        drone_states = np.zeros((self.num_drones, 6))
        # drone_states[:, 0:3] = grid_positions(
        #     num_points=self.num_drones,
        #     origin=home,
        #     space=spacing,
        # )
        drone_states[:, 0:3] = environment_random_positions(
            num_positions=self.num_drones, env=self.environment
        )
        self.drones.initialize(states=drone_states)

        # Initialize users
        user_states = np.zeros((self.num_users, 6))
        user_states[:, 0:3] = environment_random_positions(
            num_positions=self.num_users, env=self.environment
        )
        self.users.initialize(states=user_states)

        self.sdqn_brain.step()
        self.prev_frames = self.sdqn_brain.last_frames
        self.prev_actions = self.sdqn_brain.last_actions

        super().initialize()

        self.logger.info("✅ Initialization completed.")

    def update(self, dt=None) -> None:
        self.update_drone_positions()

        super().update(dt)

        self.sdqn_brain.update_positions(
            drones=self.drone_states[:, 0:2], users=self.user_states[:, 0:2]
        )

        self.rewards, self.dones = self.reward_manager.update(
            drones=self.drone_states[:, 0:2],
            users=self.user_states[:, 0:2],
            time=self.sim_time,
        )

        self.reset_collided_drones(self.dones)

        self.sdqn_brain.step()

        self.sdqn_brain.wrapper.add_experiences(
            frames=self.prev_frames,
            actions=self.prev_actions.astype(np.uint32),
            next_frames=self.sdqn_brain.last_frames,
            rewards=self.rewards.astype(np.float32),
            dones=self.dones,
        )

        self.sdqn_brain.wrapper.train()

        self.prev_frames = self.sdqn_brain.last_frames
        self.prev_actions = self.sdqn_brain.last_actions

    def reset_collided_drones(self, dones: np.ndarray) -> None:
        done_indices = np.arange(self.num_drones)[dones]
        for i in done_indices:
            state = np.zeros(6)
            state[0:3] = environment_random_positions(
                num_positions=1, env=self.environment
            )
            drone: Drone = self.drones[i]
            drone.initialize(state)

            self.logger.warning(f"⚠️  Reset drone {i} to initial states")

    def update_drone_positions(self) -> None:
        for drone, iface in zip(self.drones, self.sdqn_brain.ifaces):
            drone.state[0:2] += self.displacement * iface.direction
            drone.state[2] = self.sdqn_config.target_height

    def simulation_status_str(self) -> str:
        area_cov = self.metrics.area_coverage
        users_cov = self.metrics.user_coverage
        direct_conn = self.metrics.direct_conn
        global_conn = self.metrics.global_conn
        return (
            f"Real time: {self.real_time:.2f} s, "
            f"Sim time: {self.sim_time:.2f} s, "
            f"Sim steps: {self.sim_step}, "
            f"Area coverage: {area_cov*100.0:.2f} %, "
            f"Users coverage: {users_cov*100.0:.2f} %, "
            f"Directly connected drones: {direct_conn*100.0:.2f} %, "
            f"Globally connected drones: {global_conn*100.0:.2f} %"
        )

    def training_status_str(self) -> str:
        return self.sdqn_brain.wrapper.training_status_str()
