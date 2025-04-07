"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.extended_vsm import EVSM


class Drone:
    _counter = 0

    def __init__(self):
        super().__init__()

        Drone._counter += 1
        self.id = Drone._counter

        self.mass = 1.0
        self.max_acc = 10.0

        self.state = np.zeros(6)  # px, py, pz, vx, vy, vz
        self.control = np.zeros(3)  # ax, ay, az
        self.target_position = np.zeros(3)  # px, py, pz

        self.position_control = EVSM()

        self.visible_neighbors_ids = np.zeros((0,), dtype=np.int32)
        self.visible_neighbors_positions = np.zeros((0, 3))  # (px, py, pz)

        # self.linked_neighbors_ids = np.zeros((0,), dtype=np.int32)
        # self.linked_neighbors_positions = np.zeros((0, 3))  # (px, py, pz)

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        x_dot = np.zeros(6)
        x_dot[0:3] = state[3:6]  # dx/dt = v
        x_dot[3:6] = control  # dv/dt = a (u)
        return x_dot

    def update(self, dt: float = 0.01) -> None:
        control_force = self.position_control.update(
            self.position[0:2],
            self.velocity[0:2],
            self.visible_neighbors_positions[:, 0:2],
        )
        self.control[0:2] = control_force / self.mass
        self.control = self.limit_acceleration(self.control)
        
        x_dot = self.dynamics(self.state, self.control)
        self.state = self.state + x_dot * dt

    def set_visible_neighbors(self, ids: np.ndarray, positions: np.ndarray) -> None:
        self.visible_neighbors_ids = ids
        self.visible_neighbors_positions = positions
        
    def limit_acceleration(self, acc: np.ndarray) -> np.ndarray:
        if self.max_acc is None:
            return acc
        acc_mag = np.linalg.norm(acc)
        acc_dir = acc / acc_mag if acc_mag > 0.0 else np.zeros(3)
        return acc_dir * min(acc_mag, self.max_acc)

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6]

    @property
    def acceleration(self) -> np.ndarray:
        return self.control

    def __repr__(self) -> str:
        return f"Drone {self.id}: Position: {self.position}, Velocity: {self.velocity}"
