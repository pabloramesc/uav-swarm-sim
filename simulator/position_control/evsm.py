import numpy as np

from simulator.environment.limited_regions import LimitedRegion
from simulator.math.angles import vector_angle, is_angle_between, diff_angle_2pi
from simulator.position_control.evsm_numba import (
    calculate_links,
    calculate_control_force,
    calculate_damping_force,
    calculate_sweep_angle
)


class EVSM:
    def __init__(
        self,
        ln: float = 50.0,
        mass: float = 1.0,
        d_obs: float = 10.0,
    ) -> None:
        self.ln = ln
        self.ks = 0.2  # 1.0 / ln
        self.kd = mass / 1.0  # 1 second damping (k_d = m / tau)
        self.d_obs = d_obs
        self.k_obs = 1.0 / d_obs
        self.k_expl = 2.0 * self.ks

        self.state = np.zeros(4)  # px, py, vx, vy

        self.neighbors = np.zeros((0, 2))  # (px, py)
        self.links_mask = np.zeros((0,), dtype=np.bool)

        self.limited_regions: list[LimitedRegion] = []

        self.sweep_angles = (None, None)

    @property
    def position(self) -> np.ndarray:
        return self.state[0:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[2:4]

    def update(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        neighbors: np.ndarray,
    ) -> np.ndarray:
        self.state[0:2] = position
        self.state[2:4] = velocity
        self.neighbors = neighbors
        return self.update_from_internal()

    def update_from_internal(self) -> np.ndarray:
        # self.links_mask = self.calculate_links()
        self.links_mask = calculate_links(self.position, self.neighbors)

        # control_force = self.calculate_control_force()
        control_force = calculate_control_force(
            self.position, self.neighbors[self.links_mask], ln=self.ln, ks=self.ks
        )

        # damping_force = self.calculate_damping_force()
        damping_force = calculate_damping_force(self.velocity, kd=self.kd)

        # self.sweep_angles = self.calculate_sweep_angle()
        self.sweep_angles = calculate_sweep_angle(self.position, self.neighbors)

        exploration_force = np.zeros(2)
        if self.is_edge_robot():
            exploration_force = self.calculate_exploration_force()

        obstacles_force = self.calculate_obstacles_force()
        if self.is_near_obstacle():
            total_force = obstacles_force + damping_force
        else:
            total_force = control_force + damping_force + exploration_force

        return total_force

    def calculate_links(self) -> np.ndarray:
        num_neighbors = self.neighbors.shape[0]
        links = np.zeros((num_neighbors,), dtype=np.bool)
        for neighbor1_idx in range(num_neighbors):
            neighbor1_pos = self.neighbors[neighbor1_idx, :]
            center = (self.position + neighbor1_pos) / 2
            radius = np.linalg.norm(self.position - neighbor1_pos) / 2
            distance = np.inf
            for neighbor2_idx in range(num_neighbors):
                if neighbor2_idx == neighbor1_idx:
                    continue
                neighbor2_pos = self.neighbors[neighbor2_idx, :]
                distance = np.linalg.norm(neighbor2_pos - center)
                if distance <= radius:
                    break
            if distance > radius:
                links[neighbor1_idx] = True
        return links

    def calculate_control_force(self) -> np.ndarray:
        deltas = self.neighbors[self.links_mask] - self.position
        distances = np.linalg.norm(deltas, axis=1)[:, np.newaxis]
        directions = np.where(
            distances > 0.0, deltas / distances, np.zeros_like(deltas)
        )
        spring_forces = (distances - self.ln) * directions
        control_force = self.ks * np.sum(spring_forces, axis=0)
        return control_force

    def calculate_damping_force(self) -> np.ndarray:
        damping_force = -self.kd * self.velocity * np.linalg.norm(self.velocity)
        return damping_force

    def calculate_exploration_force(self) -> np.ndarray:
        num_regions = len(self.limited_regions)
        if num_regions == 0:
            return np.zeros(2)
        distances = np.zeros((num_regions, 1))
        directions = np.zeros((num_regions, 2))
        for i, region in enumerate(self.limited_regions):
            distances[i] = region.distance(self.position)
            directions[i] = region.direction(self.position)
        # Check if the obstacle is between the sweep angles and calculate the weights
        region_angles = vector_angle(directions)
        is_visible = is_angle_between(
            region_angles, self.sweep_angles[0], self.sweep_angles[1]
        )[:, np.newaxis]
        weights = distances / np.linalg.norm(distances) * is_visible
        # Calculate the exploration forces
        forces = weights * (distances - self.ln) * directions
        exploration_force = self.k_expl * np.sum(forces, axis=0)
        return exploration_force

    def calculate_obstacles_force(self) -> np.ndarray:
        obstacle_force = np.zeros(2)
        for region in self.limited_regions:
            distance = max(region.distance(self.position), 1e-2)
            direction = region.direction(self.position)
            obstacle_force += (self.d_obs / distance) ** 2 * (-direction)
        return obstacle_force * self.k_obs

    def calculate_sweep_angle(self) -> tuple[float, float]:
        num_neighbors = self.neighbors.shape[0]
        if num_neighbors == 0:
            return (0.0, 0.0)
        vectors = self.neighbors - self.position
        angles = vector_angle(vectors)
        angles.sort()
        for i in range(num_neighbors):
            angle1 = angles[i - 1]
            angle2 = angles[i]
            sweep_angle = diff_angle_2pi(angle2, angle1)
            if sweep_angle >= np.pi / 2:
                return (angle1, angle2)
        return (None, None)

    def is_edge_robot(self) -> bool:
        return self.sweep_angles[0] is not None and self.sweep_angles[1] is not None

    def is_near_obstacle(self) -> bool:
        for region in self.limited_regions:
            if region.distance(self.position) < self.d_obs:
                return True
        return False
