import numpy as np

from ..environment import Environment
from ..math.connectivity import (
    covered_positions,
    directly_connected,
    globally_connected,
    pairwise_connectivity_matrix,
)
from ..math.path_loss_model import signal_strength


def area_coverage(
    env: Environment,
    tx_positions: np.ndarray,
    num_points: int = 1000,
    check_obstacles: bool = False,
    tx_power: float = 20.0,
    min_rssi: float = -80.0,
    freq_mhz: float = 2412.0,
    path_loss_exp: float = 2.4,
):
    if tx_positions.shape[1] != 3:
        raise ValueError("Tx positions must be (N, 3) numpy array.")
    
    # Sample random points and compute coverage ratio
    rx_positions = np.zeros((num_points, 3))
    rx_positions[:, 0] = np.random.uniform(*env.boundary_xlim, num_points)
    rx_positions[:, 1] = np.random.uniform(*env.boundary_ylim, num_points)
    rx_positions[:, 2] = env.get_elevation(rx_positions[:, 0:2])

    valid = env.is_inside(rx_positions)

    if check_obstacles:
        valid &= ~env.is_collision(rx_positions)

    if not valid.any():
        return 0.0

    rssi = signal_strength(
        tx_positions=tx_positions,
        rx_positions=rx_positions[valid],
        f=freq_mhz,
        n=path_loss_exp,
        tx_power=tx_power,
        mode="max",
    )
    return np.mean(rssi > min_rssi)


class MetricsSnapshot:
    def __init__(
        self,
        env: Environment,
        drone_states: np.ndarray,
        user_states: np.ndarray,
        tx_power: float = 20.0,
        min_rssi: float = -80.0,
        freq_mhz: float = 2412.0,
        path_loss_exp: float = 2.4,
    ):
        """
        Snapshot of coverage and connectivity metrics for drones and users.

        Parameters
        ----------
        env : Environment
            Simulation environment instance.
        drone_states : np.ndarray
            Drone states array (N, 6) with positions in columns 0:3.
        user_states : np.ndarray
            User states array (M, 6) with positions in columns 0:3.
        tx_power : float
            Transmit power in dBm.
        min_rssi : float
            Minimum RSSI threshold to consider a link.
        freq_mhz : float
            Carrier frequency in MHz.
        path_loss_exp : float
            Path loss exponent.
        """
        self.env = env
        self.drone_states = drone_states
        self.user_states = user_states

        # Coverage
        self.area_coverage = area_coverage(
            env,
            tx_positions=self.drone_states[:, 0:3],
            num_points=1000,
            check_obstacles=True,
            tx_power=tx_power,
            min_rssi=min_rssi,
            freq_mhz=freq_mhz,
            path_loss_exp=path_loss_exp,
        )
        self.covered_users = covered_positions(
            tx_positions=self.drone_states[:, 0:3],
            rx_positions=self.user_states[:, 0:3],
            tx_power=tx_power,
            min_rssi=min_rssi,
            freq_mhz=freq_mhz,
            path_loss_exp=path_loss_exp,
        )
        self.users_coverage = len(self.covered_users) / max(len(self.user_states), 1)

        # Connectivity
        self.links_matrix = pairwise_connectivity_matrix(
            positions=self.drone_states[:, 0:3],
            tx_power=tx_power,
            min_rssi=min_rssi,
            freq_mhz=freq_mhz,
            path_loss_exp=path_loss_exp,
        )
        self.directly_connected = directly_connected(
            positions=self.drone_states[:, 0:3],
            tx_power=tx_power,
            min_rssi=min_rssi,
            freq_mhz=freq_mhz,
            path_loss_exp=path_loss_exp,
        )
        self.globally_connected = globally_connected(
            positions=self.drone_states[:, 0:3],
            tx_power=tx_power,
            min_rssi=min_rssi,
            freq_mhz=freq_mhz,
            path_loss_exp=path_loss_exp,
        )
        self.direct_connections = len(self.directly_connected) / max(
            len(self.drone_states), 1
        )
        self.global_connections = len(self.globally_connected) / max(
            len(self.drone_states), 1
        )
