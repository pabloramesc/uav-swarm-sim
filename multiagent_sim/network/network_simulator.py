"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import atexit
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..utils.logger import create_logger
from .sim_bridge import SimBridge, SimPacket

NodeType = Literal["gcs", "drone", "user"]


@dataclass
class SimNode:
    node_id: int
    node_type: NodeType
    name: str = None
    addr: str = "0.0.0.0"
    position: np.ndarray = None


class NetworkSimulator:
    NETWORK_BASE = "10.0."
    NODE_TYPE_TO_PREFIX: dict[NodeType, str] = {
        "gcs": NETWORK_BASE + "1.",
        "drone": NETWORK_BASE + "2.",
        "user": NETWORK_BASE + "3.",
    }

    def __init__(
        self, num_gcs: int, num_drones: int, num_users: int, verbose: bool = True
    ):
        self.num_gcs = num_gcs
        self.num_drones = num_drones
        self.num_users = num_users
        self._validate_number_of_nodes()

        self.nodes: list[SimNode] = []
        self.node_packets: dict[int, list[SimPacket]] = {}
        self._create_nodes()

        self.bridge = SimBridge()

        self.ns3_process = None

        self.logger = create_logger(
            "NetworkSimulator", level="INFO" if verbose else "WARNING"
        )

        atexit.register(self._kill_ns3_process)

        signal.signal(signal.SIGINT, self._on_exit_signal)
        signal.signal(signal.SIGTERM, self._on_exit_signal)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def ns3_time(self) -> float:
        if self.bridge.last_sim_time is None:
            return 0.0
        return self.bridge.last_sim_time

    def get_broadcast_address(self) -> str:
        return self.NETWORK_BASE + "255.255"

    def get_node(self, node_id: int) -> SimNode:
        self._validate_node_id(node_id)
        return self.nodes[node_id]

    def get_node_from_name(self, name: str) -> SimNode:
        for node in self.nodes:
            if node.name == name:
                return node
        raise ValueError(f"No node found with name {name}")

    def get_node_from_address(self, ip_address: str) -> SimNode:
        for node in self.nodes:
            if node.addr == ip_address:
                return node
        raise ValueError(f"No node found with IP address {ip_address}")

    def update(self, positions: np.ndarray = None, check: bool = False) -> None:
        self.fetch_packets()
        
        if positions is not None:
            self.set_node_positions(positions)
            
        if check:
            try:
                self.verify_node_positions(timeout=0.1)
            except Exception as err:
                self.logger.warning(f"⚠️  Erro verifying positions: {err}")

    def launch_simulator(self, max_attempts: int = 1) -> None:
        attempt = 1
        while attempt <= max_attempts:
            self.logger.info(
                f"➡️  Initializing NS-3 simulator... (attempt {attempt}/{max_attempts})"
            )

            try:
                self._kill_previous_ns3()
                self._launch_ns3_simulator(wait=1.0)
                self._verify_ns3_connection(max_attempts=5)
                self._verify_ns3_nodes()

                self.logger.info(
                    f"✅ NS-3 simulator successfully launched for {self.num_nodes} nodes "
                    f"({self.num_gcs} GCSs, {self.num_drones} drones, and {self.num_users} users)."
                )

                self.init_time = self.bridge.request_sim_time()
                self.real_init_time = time.time()

                return

            except Exception as err:
                self.logger.warning(f"⚠️  Launch attempt {attempt} failed: {err}")
                self._terminate_ns3_simulator()
                attempt += 1

        if attempt > max_attempts:
            raise RuntimeError("❌ All simulator launch attempts failed.")

    def shutdown_simulator(self, timeout: float = 1.0) -> None:
        self.logger.info("Terminating NS-3 simulator...")
        self.bridge.stop_simulation()
        time.sleep(timeout)
        self._terminate_ns3_simulator(timeout)
        self.ns3_process = None
        self.init_time = None
        self.real_init_time = None
        self.bridge

    def set_node_positions(self, positions: np.ndarray) -> None:
        if positions.shape != (self.num_nodes, 3):
            raise ValueError(
                f"Positions must be a numpy array with shape ({self.num_nodes}, 3)"
            )
        node_id_pos: dict[int, np.ndarray] = {}
        for node_id, node_pos in enumerate(positions):
            self.nodes[node_id].position = node_pos
            node_id_pos[node_id] = node_pos
        self.bridge.set_node_positions(node_id_pos)

    def verify_node_positions(self, timeout: float = 0.1) -> None:
        positions = self.bridge.request_node_positions(timeout)
        for node_id, ns3_pos in positions.items():
            local_pos = self.nodes[node_id].position
            if not np.allclose(local_pos, ns3_pos):
                raise Exception(
                    f"Node {node_id} local position ({local_pos}) does not match NS-3 position ({ns3_pos})"
                )

    def send_packet(self, packet: SimPacket) -> None:
        self.bridge.send_ingress_packet(packet)

    def fetch_packets(self) -> None:
        packets = self.bridge.read_egress_packets()
        for packet in packets:
            self._validate_node_id(packet.node_id)
            self.node_packets[packet.node_id].append(packet)
        return

    def get_node_packets(self, node_id: int, delete: bool = False) -> list[SimPacket]:
        self._validate_node_id(node_id)
        packets = self.node_packets[node_id]

        for packet in packets:
            if packet.node_id != node_id:
                raise ValueError(
                    f"Packet node_id {packet.node_id} does not match requested node_id {node_id}"
                )

        if delete:
            self.node_packets[node_id] = []

        return packets

    def _create_nodes(self) -> None:
        node_id = 0
        for count, node_type in (
            (self.num_gcs, "gcs"),
            (self.num_drones, "drone"),
            (self.num_users, "user"),
        ):
            prefix = self.NODE_TYPE_TO_PREFIX[node_type]
            for i in range(count):
                self.nodes.append(
                    SimNode(node_id, node_type, f"{node_type}{i}", prefix + str(i + 1))
                )
                self.node_packets[node_id] = []
                node_id += 1

    def _rewrite_ns3_code(self) -> None:
        subprocess.run(["sh", "rewrite_ns3_code.sh"], cwd="./network_sim", check=True)

    def _kill_previous_ns3(self) -> None:
        # first check if any process is running
        result = subprocess.run(
            ["pgrep", "-f", "ns3"], stdout=subprocess.PIPE, check=False
        )
        pids = result.stdout.decode("utf-8").splitlines()

        # if no match, return
        if len(pids) == 0:
            self.logger.info("No previous NS-3 process found.")
            return

        self.logger.info(
            f"Found {len(pids)} previous NS-3 process(es): {', '.join(pids)}"
        )

        # if match, kill the process
        subprocess.run(["pkill", "-f", "ns3"], check=True)
        self.logger.info("Previous NS-3 process killed.")

    def _launch_ns3_simulator(self, wait: float = 1.0) -> None:
        sim_cmd = (
            "scratch/swarm-net-sim/main "
            f"--nGCS={self.num_gcs} --nUAV={self.num_drones} --nUser={self.num_users}"
        )
        self.ns3_process = subprocess.Popen(
            ["./ns3", "run", sim_cmd], cwd="./network_sim/ns-3", preexec_fn=os.setsid
        )
        time.sleep(wait)

    def _terminate_ns3_simulator(self, timeout: float = 1.0) -> None:
        if self.ns3_process and self.ns3_process.poll() is None:
            self.logger.info("NS-3 process is still running. Terminating...")
            self.ns3_process.terminate()  # Send termination signal
            self.ns3_process.wait(timeout)  # Wait for the process to terminate
        self.logger.info("🛑 NS-3 process terminated.")

    def _kill_ns3_process(self) -> None:
        if self.ns3_process and self.ns3_process.poll() is None:
            self.logger.warning("⚠️  NS-3 process is still running. Killing...")
            pgid = os.getpgid(self.ns3_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            self.ns3_process.wait()
            self.ns3_process = None
            self.logger.info("💀 NS-3 process killed.")

    def _verify_ns3_connection(self, max_attempts: int = 2) -> None:
        self.logger.info("Verifying NS-3 connection...")
        is_running = False
        for _ in range(max_attempts):
            is_running = self.bridge.is_ns3_running()
            if is_running:
                break
        if not is_running:
            raise Exception("NS-3 simulator is not responding.")
        self.logger.info("NS-3 connection verified.")

    def _verify_ns3_nodes(self) -> None:
        self.logger.info("Verifying NS-3 nodes...")
        addresses = self.bridge.request_node_addresses()
        for node_id, node_addr in addresses.items():
            node = self.nodes[node_id]
            if node.node_id != node_id:
                raise Exception(
                    f"NS-3 node id {node_id} does not match local node id {node.node_id}"
                )
            if node.addr != node_addr:
                raise Exception(
                    f"NS-3 node addr {node_addr} does not match local node addr {node.addr}"
                )
            self._validate_node_type_address(node.node_type, node.addr)
        self.logger.info("NS-3 nodes verified.")

    def _validate_node_id(self, id: int) -> None:
        if id < 0:
            raise ValueError("Node ID must be a positive integer")
        if id >= self.num_nodes:
            raise ValueError("Node ID must be lower than the number of nodes")

    def _validate_node_type_address(self, node_type: NodeType, addr: str) -> None:
        octets = addr.split(".")
        if len(octets) != 4:
            raise ValueError(f"Address must have 4 bytes but {len(octets)} were given")

        if not all(0 <= int(octet) <= 255 for octet in octets):
            raise ValueError("Address octets must be in range 0-255")

        prefix = self.NODE_TYPE_TO_PREFIX[node_type]
        if not addr.startswith(prefix):
            raise ValueError(
                f"Node of type '{node_type}' must have address in '{prefix}x' format."
                f"But {addr} was given."
            )

    def _validate_number_of_nodes(self) -> None:
        if not 0 <= self.num_gcs <= 255:
            raise ValueError(
                f"Number of GCSs must be between 0 and 255. {self.num_gcs} was given."
            )
        if not 0 <= self.num_drones <= 255:
            raise ValueError(
                f"Number of drones must be between 0 and 255. {self.num_drones} was given."
            )
        if not 0 <= self.num_users <= 255:
            raise ValueError(
                f"Number of users must be between 0 and 255. {self.num_users} was given."
            )
        if self.num_gcs + self.num_drones + self.num_users < 1:
            raise ValueError(
                "At least one node (GCS, drone, or user) must be present in the simulation."
            )

    def _on_exit_signal(self, signum, frame):
        self.logger.warning(f"⚠️  Received signal {signum}, shutting down NS-3 ...")
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
