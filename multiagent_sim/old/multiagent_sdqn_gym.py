import time

import numpy as np

from ..agents import Agent, AgentsRegistry, ControlStation, Drone, User
from ..environment import Environment
from ..math.path_loss_model import signal_strength
from ..mobility.sdqn_position_controller import (
    SDQNConfig,
    SDQNPositionController,
)
from ..sdqn.sdqn_wrapper import SDQNWrapper
from ..sdqn.reward_manager import RewardManager
from ..sdqn.frame_generators import SimpleFrameGenerator
from ..sdqn.actions import Action
from ..sdqn.sdqn_brain import SDQNBrain
from ..sdqn.sdqn_interface import SDQNInterface
from ..mobility.utils import environment_random_positions
from ..utils.logger import create_logger


class MultiAgentSDQNGym:
    def __init__(
        self,
        num_drones: int,
        num_users: int,
        dt: float = 0.01,
        dem_path: str = None,
        sdqn_config: SDQNConfig = None,
        model_path: str = None,
    ) -> None:
        self.num_drones = num_drones
        self.num_users = num_users
        self.dt = dt
        self.environment = Environment(dem_path)
        self.sdqn_config = sdqn_config if sdqn_config else SDQNConfig()
        self.model_path = model_path

        self.sdqn_agent = self._create_sdqn_central_agent()
        self.reward_manager = self._create_reward_manager()

        self.agents = self._create_agents()

        self.init_time: float = None
        self.sim_time = 0.0
        self.sim_step = 0

        self.logger = create_logger(name="MultiAgentSDQNGym", level="INFO")

        self.prev_frames: np.ndarray = None
        self.prev_actions: np.ndarray = None

    @property
    def real_time(self) -> float:
        return time.time() - self.init_time

    def _create_sdqn_central_agent(self) -> SDQNBrain:
        frame_shape = SimpleFrameGenerator.calculate_frame_shape()
        num_actions = 5  # len(Action)
        wrapper = SDQNWrapper(
            frame_shape, num_actions, model_path=self.model_path, train_mode=True
        )
        return SDQNBrain(wrapper)

    def _create_reward_manager(self) -> RewardManager:
        manager = RewardManager(env=self.environment)
        return manager

    def _create_agents(self) -> list[Agent]:
        agents: list[Agent] = []

        self.gcs = ControlStation(agent_id=len(agents), environment=self.environment)
        agents.append(self.gcs)

        self.drones = AgentsRegistry()
        self.users = AgentsRegistry()

        for i in range(self.num_drones):
            frame_gen = SimpleFrameGenerator(env=self.environment, frame_radius=250.0)
            local = SDQNInterface(iface_id=i, frame_generator=frame_gen)
            self.sdqn_agent.register_interface(local)
            sdqn = SDQNPositionController(
                config=self.sdqn_config, environment=self.environment, sdqn_iface=local
            )
            drone = Drone(
                agent_id=len(agents),
                environment=self.environment,
                position_controller=sdqn,
                drones_registry=self.drones,
                users_registry=self.users,
                neighbor_provider="registry",
            )
            self.drones.register(drone)
            agents.append(drone)

        for _ in range(self.num_users):
            user = User(agent_id=len(agents), environment=self.environment)
            self.users.register(user)
            agents.append(user)

        return agents

    def initialize(self) -> None:
        self.logger.info("Initializing simulation ...")

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

        self.init_time = time.time()
        self.sim_time = 0.0
        self.sim_steps = 0

        self.sdqn_agent.step()
        self.prev_frames = self.sdqn_agent.last_frames
        self.prev_actions = self.sdqn_agent.last_actions

        self.sim_time = 0.0
        self.sim_steps = 0
        self.init_time = time.time()

        self.logger.info("✅ Initialization completed.")

    def update(self, dt: float = None) -> None:
        dt = dt if dt is not None else self.dt
        self.sim_time += dt
        self.sim_steps += 1

        self.gcs.update(dt)
        self.drones.update(dt)
        self.users.update(dt)

        self.drone_states = self.drones.get_states_array()
        self.user_states = self.users.get_states_array()

        self.rewards, dones = self.reward_manager.update(
            drones=self.drone_states[:, 0:2],
            users=self.user_states[:, 0:2],
            time=self.sim_time,
        )
        self.reset_collided_drones(dones)

        self.sdqn_agent.step()

        self.sdqn_agent.sdqn.add_experiences(
            frames=self.prev_frames,
            actions=self.prev_actions,
            next_frames=self.sdqn_agent.last_frames,
            rewards=self.rewards,
            dones=dones,
        )

        self.sdqn_agent.sdqn.train()

        self.prev_frames = self.sdqn_agent.last_frames
        self.prev_actions = self.sdqn_agent.last_actions

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

    def area_coverage(self, num_points: int = 1000, rx_sens: float = -80.0) -> float:
        eval_points = np.zeros((num_points, 3))
        eval_points[:, 0] = np.random.uniform(
            *self.environment.boundary_xlim, num_points
        )
        eval_points[:, 1] = np.random.uniform(
            *self.environment.boundary_ylim, num_points
        )
        eval_points[:, 2] = self.environment.get_elevation(eval_points[:, 0:2])
        in_area = self.environment.is_inside(
            eval_points
        ) & ~self.environment.is_collision(eval_points, check_altitude=False)
        if not np.any(in_area):
            return 0.0
        tx_power = signal_strength(
            tx_positions=self.drone_states[:, 0:3],
            rx_positions=eval_points[in_area],
            f=2412,
            n=2.4,
            tx_power=20.0,
            mode="max",
        )
        in_range = tx_power > rx_sens
        return np.sum(in_range) / np.sum(in_area)

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

    def _needs_update(self) -> bool:
        if self.sim_time is None or self.last_update_time is None:
            raise Exception("Bad time initialization.")
        elapsed_time = self.sim_time - self.last_update_time
        return elapsed_time > self.update_period
