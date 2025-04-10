import numpy as np

from simulator.environment.limited_regions import LimitedRegion
from simulator.math.angles import (
    SweepAngle,
    diff_angle_2pi,
    is_angle_between,
    vector_angle,
)
from simulator.position_control.evsm_numba import (
    calculate_avoidance_force,
    calculate_control_force,
    calculate_damping_force,
    calculate_links,
    calculate_sweep_angle,
    calculate_exploration_force,
    compile_all
)


class EVSM:
    _compiled = False
    def __init__(
        self,
        ln: float = 50.0,
        mass: float = 1.0,
        d_obs: float = 10.0,
    ) -> None:
        self.ln = ln
        self.ks = 1.0 / ln
        self.kd = mass / 1.0  # 1 second damping (k_d = m / tau)
        self.d_obs = d_obs
        self.k_obs = 1.0 / d_obs
        self.k_expl = 2.0 * self.ks

        self.state = np.zeros(4)  # px, py, vx, vy

        self.neighbors = np.zeros((0, 2))  # (px, py)
        self.links_mask = np.zeros((0,), dtype=bool)

        self.avoid_regions: list[LimitedRegion] = []

        self.sweep_angle: SweepAngle = None
        
        # if not EVSM._compiled:
        #     print("Compiling EVSM numba functions ...")
        #     compile_all()
        #     print("✅ EVSM numba functions compiled.")
        #     EVSM._compiled = True

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
        self.state[0:2] = position.copy()
        self.state[2:4] = velocity.copy()
        self.neighbors = neighbors.copy()
        return self.update_from_internal()

    def update_from_internal(self) -> np.ndarray:
        self.links_mask = calculate_links(self.position, self.neighbors)

        control_force = calculate_control_force(
            self.position, self.neighbors[self.links_mask], ln=self.ln, ks=self.ks
        )

        # damping_force = self.calculate_damping_force()
        damping_force = calculate_damping_force(self.velocity, kd=self.kd)

        self.sweep_angle = self._calculate_sweep_angle()

        region_distances, region_directions = (
            self.get_avoidance_distances_and_directions()
        )

        exploration_force = np.zeros(2)
        if self.is_edge_robot():
            exploration_force = calculate_exploration_force(
                region_distances,
                region_directions,
                self.sweep_angle.to_tuple(),
                ln=self.ln,
                ks=self.k_expl,
            )

        obstacles_force = calculate_avoidance_force(
            region_distances, region_directions, d_min=self.d_obs, ks=self.k_obs
        )
        if self.is_near_obstacle():
            total_force = obstacles_force + damping_force
        else:
            total_force = control_force + damping_force + exploration_force

        return total_force

    def _calculate_sweep_angle(self) -> SweepAngle:
        start, stop = calculate_sweep_angle(self.position, self.neighbors)
        if np.isnan((start, stop)).any():
            return None
        return SweepAngle(start, stop)
    
    def set_natural_length(self, ln: float) -> None:
        if ln <= 0.0:
            raise ValueError("Natural lenght must be greater than 0.0")
        self.ln = ln

    def get_avoidance_distances_and_directions(self) -> tuple[np.ndarray, np.ndarray]:
        num_regions = len(self.avoid_regions)
        distances = np.zeros((num_regions,))
        directions = np.zeros((num_regions, 2))
        for i, region in enumerate(self.avoid_regions):
            distances[i] = region.distance(self.position)
            directions[i, :] = region.direction(self.position)
        return distances, directions

    def is_edge_robot(self) -> bool:
        return self.sweep_angle is not None

    def is_near_obstacle(self) -> bool:
        for region in self.avoid_regions:
            if region.distance(self.position) < self.d_obs:
                return True
        return False
