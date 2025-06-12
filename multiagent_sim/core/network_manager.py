import numpy as np

from ..agents import Agent
from ..network import NetworkSimulator


class NetworkManager:

    def __init__(self, num_gcs: int, num_drones: int, num_users: int):
        self.netsim = NetworkSimulator(
            num_gcs,
            num_drones,
            num_users,
            verbose=True,
        )

        self.update_period = 0.1
        self.checking_period = 1.0

        self._last_update_time: float = None
        self._last_checking_time: float = None

    @property
    def ns3_time(self) -> float:
        return self.netsim.ns3_time

    def initialize(self, positions: dict[int, np.ndarray]) -> None:
        self.netsim.launch_simulator(max_attempts=2)
        self.netsim.set_node_positions(positions)
        self.netsim.verify_node_positions()

    def update(self, time: float, positions: dict[int, np.ndarray]) -> None:
        if self._needs_update(time):
            agent_positions = positions
            self._last_update_time = time
        else:
            agent_positions = None

        if self._needs_cheking(time):
            check = True
            self._last_checking_time = time
        else:
            check = False

        self.netsim.update(agent_positions, check)

    def wait(self, timeout: float) -> None:
        self.netsim.fetch_packets()
        try:
            self.netsim.bridge.request_sim_time(timeout)
        except TimeoutError:
            pass

    def _needs_update(self, time: float) -> bool:
        if self._last_update_time is None:
            return True
        return time >= self._last_update_time + self.update_period

    def _needs_cheking(self, time: float) -> bool:
        if self._last_checking_time is None:
            return True
        return time >= self._last_checking_time + self.checking_period
