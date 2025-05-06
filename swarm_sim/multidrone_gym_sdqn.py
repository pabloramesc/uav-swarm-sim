import time

import numpy as np

from .agents.drone import Drone
from .environment import Environment
from .math.path_loss_model import signal_strength
from .position_control.sdqn_position_control import SDQNConfig, SDQNPostionController
from .sdqn.central_agent import CentralAgent
from .sdqn.reward_manager import RewardManager


class MultidroneGymSDQN:
    def __init__(
        self,
        num_drones: int,
        dt: float = 0.01,
        config: SDQNConfig = None,
        model_path: str = None,
        verbose: bool = True,
        train: bool = True,
    ) -> None:
        self.num_drones = num_drones
        self.config = config or SDQNConfig()
        self.dt = dt
        self.verbose = verbose
        self.update_period = 0.1

        self.environment = Environment()

        self.drones: list[Drone] = []
        for id in range(self.num_drones):
            controller = SDQNPostionController(self.config, self.environment)
            drone = Drone(id, self.environment, controller)
            self.drones.append(drone)

        self.drone_states = np.zeros((self.num_drones, 6))  # px, py, pz, vx, vy, vz
        self.initial_states = np.zeros((self.num_drones, 6))

        self.central_agent = CentralAgent(
            num_drones=self.num_drones,
            num_cells=self.config.num_cells,
            num_channels=controller.dqns.frame_shape[-1],
            training_mode=train,
            model_path=model_path,
        )
        self.model_path = self.central_agent.model_path
        
        self.reward_manager = RewardManager(self.environment)

        self.real_t0: float = None
        self.sim_time: float = None
        self.sim_steps: int = None

        self.last_update_time: float = None
        
        self.prev_actions: np.ndarray = None
        self.prev_frames: np.ndarray = None
        
        self.frames: np.ndarray = None
        self.actions: np.ndarray = None
        self.rewards: np.ndarray = None
        self.dones: np.ndarray = None

    @property
    def real_time(self) -> float:
        return time.time() - self.real_t0

    @property
    def drone_positions(self) -> np.ndarray:
        """
        A (N, 3) shape array with drone positions [px, py, pz] in meters,
        where N is the number of drones.
        """
        return self.drone_states[:, 0:3]

    @property
    def drone_velocities(self) -> np.ndarray:
        """
        A (N, 3) shape array with drone velocities [vx, vy, vz] in m/s,
        where N is the number of drones.
        """
        return self.drone_states[:, 3:6]

    def initialize(self) -> None:
        self.drone_states = np.zeros((self.num_drones, 6))
        self.initial_states = np.zeros((self.num_drones, 6))

        for i in range(self.num_drones):
            self.initial_states[i, 0:2] = self._generate_random_position()

        for i, drone in enumerate(self.drones):
            indices = np.arange(self.num_drones)
            neighbor_indices = indices[indices != i]
            drone.initialize(
                state=self.initial_states[i, :],
                neighbor_states=self.initial_states[neighbor_indices, :],
                neighbor_ids=neighbor_indices,
                time=0.0,
            )

        self._get_drone_states()
        self.initial_states = np.copy(self.drone_states)

        self.prev_frames = self.compute_frames()
        self.prev_actions = self.compute_actions(self.prev_frames)
        self.set_target_positions(self.prev_actions)
        self.frames = self.prev_frames
        self.actions = self.prev_actions
        self.rewards = np.zeros(self.num_drones)
        self.dones = np.zeros(self.num_drones, dtype=bool)

        self.sim_time = 0.0
        self.sim_steps = 0
        self.real_t0 = time.time()
        self.last_update_time = 0.0

    def update(self, dt: float = None) -> None:
        dt = dt if dt is not None else self.dt
        self.sim_time += dt
        self.sim_steps += 1

        for drone in self.drones:
            drone.update(dt)
        self._get_drone_states()
        self._set_neighbors()

        if not self._needs_update():
            return None

        self.frames = self.compute_frames()
        self.actions = self.compute_actions(self.frames)
        self.set_target_positions(self.actions)

        self.rewards, self.dones = self.reward_manager.update(
            self.drone_positions, self.sim_time
        )
        self.reset_collided_drones(self.dones)

        self.central_agent.add_experiences(
            states=self.prev_frames,
            next_states=self.frames,
            actions=self.prev_actions,
            rewards=self.rewards,
            dones=self.dones,
        )

        self.central_agent.train()

        self.prev_frames = self.frames
        self.prev_actions = self.actions
        self.last_update_time = self.sim_time

    def compute_frames(self) -> np.ndarray:
        frames = np.zeros(self.central_agent.states_shape, dtype=np.uint8)
        for i, drone in enumerate(self.drones):
            dqns: SDQNPostionController = self._get_drone_position_controller(drone)
            frame = dqns.get_frame()
            frames[i] = frame
        return frames

    def compute_actions(self, frames: np.ndarray) -> np.ndarray:
        actions = self.central_agent.act(frames)
        return actions

    def set_target_positions(self, actions: np.ndarray) -> None:
        for i, drone in enumerate(self.drones):
            dqns: SDQNPostionController = self._get_drone_position_controller(drone)
            dqns.set_target_position(actions[i])

    def reset_collided_drones(self, dones: np.ndarray) -> None:
        done_indices = np.arange(self.num_drones)[dones]
        for i in done_indices:
            # Reset initial states to random position inside boundary
            self.initial_states[i, :] = 0.0
            self.initial_states[i, 0:2] = self._generate_random_position()

            # Get neighbors indices (same as ids)
            indices = np.arange(self.num_drones)
            neighbor_ids = indices[indices != i]

            # Initialize collided drone
            drone: Drone = self.drones[i]
            drone.initialize(
                state=self.initial_states[i, :],
                neighbor_states=self.drone_states[neighbor_ids],
                neighbor_ids=neighbor_ids,
            )

        if self.verbose and np.any(dones):
            print("⚠️ Reset drones to initial states:", done_indices)

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
            self.drone_positions, eval_points[in_area], f=2.4e3, mode="max"
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
            f"Train steps: {self.central_agent.train_steps}, "
            f"Train speed: {self.central_agent.train_speed:.2f} sps, "
            f"Memory size: {self.central_agent.memory_size}, "
            f"Epsilon: {self.central_agent.epsilon:.4f}, "
            f"Loss: {self.central_agent.loss:.4e}, "
            f"Accuracy: {self.central_agent.accuracy*100:.2f} %"
        )

    def _get_drone_states(self) -> None:
        """
        Updates the `drone_states` array with the current states of all drones.
        """
        for i, drone in enumerate(self.drones):
            self.drone_states[i, 0:3] = drone.position
            self.drone_states[i, 3:6] = drone.velocity

    def _set_drone_states(self) -> None:
        """
        Updates the states of all drones with the values in the `drone_states` array.
        """
        for i, drone in enumerate(self.drones):
            drone.state[0:3] = self.drone_states[i, 0:3]
            drone.state[3:6] = self.drone_states[i, 3:6]

    def _set_neighbors(self) -> None:
        for i, drone in enumerate(self.drones):
            indices = np.arange(self.num_drones)
            neighbor_ids = indices[indices != i]
            neighbor_states = self.drone_states[neighbor_ids]
            drone.set_neighbors(neighbor_ids, neighbor_states)

    def _needs_update(self) -> bool:
        if self.sim_time is None or self.last_update_time is None:
            raise Exception("Bad time initialization.")
        elapsed_time = self.sim_time - self.last_update_time
        return elapsed_time > self.update_period

    def _get_drone_position_controller(self, drone: Drone) -> SDQNPostionController:
        if not isinstance(drone.position_controller, SDQNPostionController):
            raise Exception(f"Drone {drone.id} position controller is not DQNS")
        return drone.position_controller

    def _generate_random_position(self) -> np.ndarray:
        px = np.random.uniform(*self.environment.boundary_xlim)
        py = np.random.uniform(*self.environment.boundary_ylim)
        return np.array([px, py])
