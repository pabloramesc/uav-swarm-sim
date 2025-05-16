import numpy as np

from ..agents import Drone
from ..mobility.sdqn_position_controller import SDQNConfig, SDQNPositionController
from ..sdqn import (
    RewardManager,
    SDQNBrain,
    SDQNWrapper,
    SimpleFrameGenerator,
    SDQNInterface,
)
from .multiagent_simulator import MultiAgentSimulator
from ..mobility.utils import environment_random_positions


class SDQNTrainer(MultiAgentSimulator):

    def __init__(
        self,
        num_drones: int,
        num_users: int = 0,
        dt: float = 0.01,
        dem_path: str = None,
        use_network: bool = False,
        sdqn_config: SDQNConfig = None,
        model_path: str = None,
    ) -> None:
        self.sdqn_config = sdqn_config or SDQNConfig()
        self.model_path = model_path
        self.sdqn_brain = self._create_sdqn_brain()

        super().__init__(
            num_drones,
            num_users,
            dt,
            dem_path,
            use_network,
            sdqn_config=self.sdqn_config,
        )

        self.reward_manager = RewardManager(env=self.environment)
        
        self.prev_frames: np.ndarray = None
        self.prev_actions: np.ndarray = None

    def _create_sdqn_brain(self) -> SDQNBrain:
        frame_shape = SimpleFrameGenerator.calculate_frame_shape()
        num_actions = 5  # len(Action)
        wrapper = SDQNWrapper(
            frame_shape, num_actions, model_path=self.model_path, train_mode=True
        )
        return SDQNBrain(wrapper)

    def _create_sdqn_interface(self, iface_id: int) -> SDQNInterface:
        frame_gen = SimpleFrameGenerator(env=self.environment, frame_radius=250.0)
        interface = SDQNInterface(iface_id, frame_gen)
        self.sdqn_brain.register_interface(interface)
        return interface

    def _create_drone(self, sdqn_config: SDQNConfig = None, **kwargs) -> Drone:
        iface = self._create_sdqn_interface(iface_id=len(self.agents))
        sdqn = SDQNPositionController(
            config=sdqn_config, env=self.environment, sdqn_iface=iface
        )
        drone = Drone(
            agent_id=len(self.agents),
            environment=self.environment,
            position_controller=sdqn,
            network_sim=self.network_simulator,
            drones_registry=self.drones,
            users_registry=self.users,
            neighbor_provider="network" if self.network else "registry",
        )
    
    def initialize(self) -> None:
        self.logger.info("Initializing simulation ...")
        
        super().initialize()
        
        self.gcs.initialize(state=np.zeros(6))

        drone_states = np.zeros((self.num_drones, 6))
        drone_states[:, 0:3] = environment_random_positions(
            num_positions=self.num_drones, env=self.environment
        )
        self.drones.initialize(states=drone_states)

        user_states = np.zeros((self.num_users, 6))
        user_states[:, 0:3] = environment_random_positions(
            num_positions=self.num_users, env=self.environment
        )
        self.users.initialize(states=user_states)
        
        self.sdqn_brain.step()
        self.prev_frames = self.sdqn_brain.last_frames
        self.prev_actions = self.sdqn_brain.last_actions

        self.logger.info("✅ Initialization completed.")
        
    def update(self, dt = None) -> None:
        super().update(dt)

        self.drone_states = self.drones.get_states_array()
        self.user_states = self.users.get_states_array()

        self.rewards, self.dones = self.reward_manager.update(
            drones=self.drone_states[:, 0:2],
            users=self.user_states[:, 0:2],
            time=self.sim_time,
        )
        self.reset_collided_drones(self.dones)

        self.sdqn_brain.step()

        self.sdqn_brain.sdqn.add_experiences(
            frames=self.prev_frames,
            actions=self.prev_actions,
            next_frames=self.sdqn_brain.last_frames,
            rewards=self.rewards,
            dones=self.dones,
        )

        self.sdqn_brain.sdqn.train()

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

    def simulation_status_str(self) -> str:
        return (
            f"Real time: {self.real_time:.2f} s, "
            f"Sim time: {self.sim_time:.2f} s, "
            f"Sim steps: {self.sim_steps}, "
            f"Area coverage: {self.area_coverage()*100:.2f} %"
        )

    def training_status_str(self) -> str:
        return (
            f"Train steps: {self.sdqn_agent.sdqn.train_steps}, "
            f"Train speed: {self.sdqn_agent.sdqn.train_speed:.2f} sps, "
            f"Memory size: {self.sdqn_agent.sdqn.memory_size}, "
            f"Epsilon: {self.sdqn_agent.sdqn.epsilon:.4f}, "
            f"Loss: {self.sdqn_agent.sdqn.loss:.4e}, "
            f"Accuracy: {self.sdqn_agent.sdqn.accuracy*100:.2f} %"
        )