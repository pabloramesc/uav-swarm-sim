import numpy as np
from .environment import Environment
from .agents import Drone
from .swarming import DQNSAgent
from .position_control import DQNSPostionController, DQNSConfig, PositionController
from .math.path_loss_model import signal_strength


class MultidroneGymDQNS:
    def __init__(
        self,
        num_drones: int,
        dt: float = 0.01,
        visible_distance: float = 100.0,
    ) -> None:
        self.num_drones = num_drones
        self.dt = dt
        self.visible_distance = visible_distance

        self.environment = Environment()

        self.time = 0.0
        self.step = 0

        self.config = DQNSConfig()

        self.drones: list[Drone] = []
        for id in range(self.num_drones):
            dqns = DQNSPostionController(self.config, self.environment)
            drone = Drone(id, self.environment, dqns)
            self.drones.append(drone)

        self.drone_states = np.zeros(
            (self.num_drones, 6), dtype=np.float32
        )  # px, py, pz, vx, vy, vz
        self.initial_states = np.zeros((self.num_drones, 6), dtype=np.float32)

        self.dqns_agent = DQNSAgent(
            self.num_drones, training_mode=True, model_path="dqns-model-01.keras"
        )
        self.last_update_time: float = None
        self.prev_states: np.ndarray = None

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

    def initialize(self, positions: np.ndarray = None) -> None:
        if positions is not None:
            self.drone_states[:, 0:3] = positions  # Fix assignment to update all drones
            self._set_drone_states()
            self.initial_states = np.copy(self.drone_states)

    def update(self, dt: float = None) -> dict:
        dt = dt if dt is not None else self.dt
        self.time += dt
        self.step += 1

        for drone in self.drones:
            indices = np.arange(self.num_drones)
            indices = indices[indices != drone.id]
            drone.set_neighbors(ids=indices, states=self.drone_states[indices])

        states = np.zeros(self.dqns_agent.states_shape, dtype=np.uint8)
        for i, drone in enumerate(self.drones):
            dqns: DQNSPostionController = self._get_drone_position_controller(drone)
            frame = dqns.get_frame()
            states[i] = frame

        actions = self.dqns_agent.act(states)
        for i, drone in enumerate(self.drones):
            dqns: DQNSPostionController = self._get_drone_position_controller(drone)
            dqns.set_target_position(actions[i])

        for drone in self.drones:
            drone.update(dt)

        rewards, dones = self.calculate_rewards_and_dones()

        if self.prev_states is not None:
            self.dqns_agent.add_experiences(
                states=self.prev_states,
                next_states=states,
                actions=actions,
                rewards=rewards,
                dones=dones,
            )

        metrics = self.dqns_agent.train()

        self.prev_states = states

        return metrics

    def calculate_rewards_and_dones(self) -> tuple[np.ndarray, np.ndarray]:
        global_score = self.area_coverage_ratio()**2
        rewards = np.ones(self.num_drones, dtype=np.float32) * global_score
        dones = np.zeros(self.num_drones, dtype=bool)
        for i, drone in enumerate(self.drones):
            inside = self.environment.is_inside(drone.position)
            collision = self.environment.is_collision(
                drone.position, check_altitude=False
            )
            terminated = (not inside) or collision
            dones[i] = terminated
            if terminated:
                rewards[i] = -1.0
                drone.state = self.initial_states[i]
        return rewards, dones

    def area_coverage_ratio(
        self, num_points: int = 1000, rx_sens: float = -80.0
    ) -> float:
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
        ) & ~self.environment.is_collision(eval_points)
        tx_power = signal_strength(
            self.drone_positions, eval_points[in_area], f=2.4e3, mode="max"
        )
        in_range = tx_power > rx_sens
        return np.sum(in_range) / np.sum(in_area)

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

    def _needs_update(self, time: float) -> bool:
        if time is None or self.last_update_time is None:
            return True
        elapsed_time = time - self.last_update_time
        return elapsed_time > 1.0

    def _get_drone_position_controller(self, drone: Drone) -> DQNSPostionController:
        if not isinstance(drone.position_controller, DQNSPostionController):
            raise Exception(f"Drone {drone.id} position controller is not DQNS")
        return drone.position_controller
