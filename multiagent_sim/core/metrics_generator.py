import numpy as np

from ..environment import Environment
from ..math.path_loss_model import signal_strength
from ..network import NetworkSimulator


class MetricsGenerator:
    def __init__(self, env: Environment, netsim: NetworkSimulator = None):
        self.env = env
        self.netsim = netsim

    def area_coverage(
        self,
        drone_positions: np.ndarray,
        num_points: int = 1000,
        min_rssi: float = -80.0,
        check_obstacles: bool = False,
    ):
        # sample random points and compute coverage ratio
        pts = np.zeros((num_points, 3))
        pts[:, 0] = np.random.uniform(*self.env.boundary_xlim, num_points)
        pts[:, 1] = np.random.uniform(*self.env.boundary_ylim, num_points)
        pts[:, 2] = self.env.get_elevation(pts[:, 0:2])

        if check_obstacles:
            inside = self.env.is_inside(pts) & ~self.env.is_collision(pts)
        else:
            inside = self.env.is_inside(pts)

        if not inside.any():
            return 0.0

        tx = drone_positions
        rx = pts[inside]
        rssi = signal_strength(tx, rx, f=2412, n=2.4, tx_power=20.0, mode="max")
        return np.mean(rssi > min_rssi)
