from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..agents import Drone
from ..mobility.sdqn_position_controller import SDQNConfig, SDQNPositionController
from ..mobility.swarm_position_controller import DummyPositionController
from ..sdqn import (
    RewardManager,
    SDQNBrain,
    SDQNWrapper,
    GridFrameGenerator,
    LogPolarFrameGenerator,
    SDQNInterface,
)
from .multiagent_simulator import MultiAgentSimulator
from ..mobility.utils import environment_random_positions, grid_positions

ActionsMode = Literal["basic", "extended"]


class SDQNSimulator(MultiAgentSimulator):

    def __init__(
        self,
        num_drones: int,
        num_users: int = 0,
        dt: float = 0.01,
        dem_path: str = None,
        use_network: bool = False,
        sdqn_config: SDQNConfig = None,
        model_path: str = None,
        logpolar: bool = False,
        actions_mode: ActionsMode = "basic",
    ) -> None:
        self.sdqn_config = sdqn_config or SDQNConfig()
        self.model_path = model_path
        self.logpolar = logpolar

        if actions_mode == "basic":
            self.num_actions = 5
        elif actions_mode == "extended":
            self.num_actions = 7
        else:
            raise ValueError("Invalid actions mode:", actions_mode)

        self.sdqn_brain = self._create_sdqn_brain()

        super().__init__(
            num_drones=num_drones,
            num_users=num_users,
            num_gcs=0,
            dt=dt,
            dem_path=dem_path,
            use_network=use_network,
        )

        self.reward_manager = RewardManager(env=self.environment)

        self.prev_frames: np.ndarray = None
        self.prev_actions: np.ndarray = None

        self.sdqn_update_period: float = 0.1
        self.last_sdqn_update_time: float = None

    def _create_sdqn_brain(self) -> SDQNBrain:
        if self.logpolar:
            frame_shape = LogPolarFrameGenerator.calculate_frame_shape()
        else:
            frame_shape = GridFrameGenerator.calculate_frame_shape()
        wrapper = SDQNWrapper(
            frame_shape=frame_shape,
            num_actions=self.num_actions,
            model_path=self.model_path,
            train_mode=False,
        )
        return SDQNBrain(wrapper)

    def _create_sdqn_interface(self, iface_id: int) -> SDQNInterface:
        if self.logpolar:
            frame_gen = LogPolarFrameGenerator(env=self.environment)
        else:
            frame_gen = GridFrameGenerator(env=self.environment, frame_radius=500.0)
        interface = SDQNInterface(iface_id, frame_gen)
        self.sdqn_brain.register_interface(interface)
        return interface

    def _create_drone(self, sdqn_config: SDQNConfig = None, **kwargs) -> Drone:
        iface = self._create_sdqn_interface(iface_id=len(self.agents))
        dummy_controller = DummyPositionController(
            config=self.sdqn_config, env=self.environment
        )
        # sdqn = SDQNPositionController(
        #     config=sdqn_config, environment=self.environment, sdqn_iface=iface
        # )
        drone = Drone(
            agent_id=len(self.agents),
            environment=self.environment,
            position_controller=dummy_controller,
            network_sim=self.netsim,
            drones_registry=self.drones,
            users_registry=self.users,
            neighbor_provider="network" if self.network else "registry",
        )
        return drone

    def initialize(self, home: ArrayLike = [0.0, 0.0], spacing: float = 5.0) -> None:
        self.logger.info("Initializing simulation ...")

        gcs_states = np.zeros((1,6))
        gcs_states[0, 0:2] = np.asarray(home[0:2])
        gcs_states[0, 2] = self.environment.get_elevation(home[0:2])
        self.gcs.initialize(states=gcs_states)

        drone_states = np.zeros((self.num_drones, 6))
        drone_states[:, 0:3] = grid_positions(
            num_points=self.num_drones,
            origin=home,
            space=spacing,
        )
        # drone_states[:, 0:3] = environment_random_positions(
        #     num_positions=self.num_drones, env=self.environment
        # )
        self.drones.initialize(states=drone_states)

        user_states = np.zeros((self.num_users, 6))
        user_states[:, 0:3] = environment_random_positions(
            num_positions=self.num_users, env=self.environment
        )
        self.users.initialize(states=user_states)

        self.sdqn_brain.step()
        self.last_sdqn_update_time = None

        super().initialize()

        self.logger.info("✅ Initialization completed.")

    def update(self, dt=None) -> None:
        self.update_drone_positions()
        
        super().update(dt)

        if not self._need_update_sdqn(self.sim_time):
            return
        self.last_sdqn_update_time = self.sim_time

        self.rewards, self.dones = self.reward_manager.update(
            drones=self.drone_states[:, 0:2],
            users=self.user_states[:, 0:2],
            time=self.sim_time,
        )

        self.reset_collided_drones(self.dones)

        self.sdqn_brain.step()

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

    def simulation_status_str(self) -> str:
        coverage = self.metrics.area_coverage(self.drone_states[:, 0:3])
        return (
            f"Real time: {self.real_time:.2f} s, "
            f"Sim time: {self.sim_time:.2f} s, "
            f"Sim steps: {self.sim_step}, "
            f"Area coverage: {coverage*100.0:.2f} %"
        )

    def training_status_str(self) -> str:
        return self.sdqn_brain.wrapper.training_status_str()

    def _need_update_sdqn(self, time: float) -> bool:
        if self.last_sdqn_update_time is None:
            return True
        return (time - self.last_sdqn_update_time) >= self.sdqn_update_period
