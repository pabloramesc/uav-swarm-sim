"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .sim_bridge import SimBridge, SimPacket


class NodeType(StrEnum):
    GCS = "gcs"
    DRONE = "drone"
    USER = "user"


@dataclass
class SimNode:
    node_id: int
    node_type: NodeType
    name: str
    addr: str
    position: np.ndarray


logger = logging.getLogger(__name__)


class NetworkSimulator:
    MAX_DATAGRAM_SIZE = 1_024
    POSITION_ENTRY_SIZE = 13
    MAX_NODES = (MAX_DATAGRAM_SIZE - 1) // POSITION_ENTRY_SIZE

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _NS3_ROOT = _PROJECT_ROOT / "ns3" / "ns-3"

    network_base = "10.0."
    node_type_to_prefix: dict[NodeType, str] = {
        NodeType.GCS: network_base + "1.",
        NodeType.DRONE: network_base + "2.",
        NodeType.USER: network_base + "3.",
    }

    def __init__(self, num_gcs: int, num_drones: int, num_users: int):
        self.num_gcs = num_gcs
        self.num_drones = num_drones
        self.num_users = num_users
        self._validate_number_of_nodes()

        self.nodes: list[SimNode] = []
        self.node_packets: dict[int, list[SimPacket]] = {}
        self._create_nodes()

        self.bridge = SimBridge()

        self.ns3_process = None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def ns3_time(self) -> float:
        if self.bridge.last_ns3_time is None:
            return 0.0
        return self.bridge.last_ns3_time

    def get_broadcast_address(self) -> str:
        return self.network_base + "255.255"

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

    def update(
        self,
        positions: dict[int, NDArray[np.float64]] | None = None,
        check: bool = False,
    ) -> None:
        self.fetch_packets()

        if positions is not None:
            self.set_node_positions(positions)

        if check:
            try:
                self.verify_node_positions(timeout=0.1)
            except Exception as err:
                logger.warning("Error verifying ns-3 positions: %s", err)

    def launch_simulator(self, max_attempts: int = 1) -> None:
        attempt = 1
        while attempt <= max_attempts:
            logger.info(
                "Initializing ns-3 simulator (attempt %d/%d).",
                attempt,
                max_attempts,
            )

            try:
                self._terminate_ns3_simulator()
                self._launch_ns3_simulator(wait=1.0)
                self._verify_ns3_connection(max_attempts=5)
                self._verify_ns3_nodes()

                logger.info(
                    "Launched ns-3 for %d nodes (%d GCS, %d drones, %d users).",
                    self.num_nodes,
                    self.num_gcs,
                    self.num_drones,
                    self.num_users,
                )

                self.init_time = self.bridge.request_ns3_time()
                self.real_init_time = time.time()

                return

            except Exception as err:
                logger.warning("ns-3 launch attempt %d failed: %s", attempt, err)
                self._terminate_ns3_simulator()
                attempt += 1

        raise RuntimeError("All ns-3 simulator launch attempts failed.")

    def shutdown_simulator(self, timeout: float = 1.0) -> None:
        logger.info("Terminating ns-3 simulator...")
        self.bridge.stop_simulation()
        time.sleep(timeout)
        self._terminate_ns3_simulator(timeout)
        self.ns3_process = None
        self.init_time = None
        self.real_init_time = None

    def set_node_positions(self, positions: dict[int, NDArray[np.float64]]) -> None:
        validated: dict[int, NDArray[np.float64]] = {}
        for node_id, position in positions.items():
            self._validate_node_id(node_id)
            value = np.asarray(position, dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Positions must be finite arrays with shape (3,).")
            validated[int(node_id)] = value.copy()

        for node_id, position in validated.items():
            self.nodes[node_id].position = position
        self.bridge.set_node_positions(validated)

    def reset(self) -> None:
        """Clear episode-local packet buffers and bridge state."""

        for packets in self.node_packets.values():
            packets.clear()
        self.bridge.reset()

    def verify_node_positions(self, timeout: float = 0.1) -> None:
        positions = self.bridge.request_node_positions(timeout)
        self._validate_complete_node_set(positions)
        for node_id, ns3_pos in positions.items():
            self._validate_node_id(node_id)
            local_pos = self.nodes[node_id].position
            if not np.allclose(local_pos, ns3_pos, atol=1.0):
                error = np.linalg.norm(local_pos - ns3_pos)
                raise RuntimeError(
                    f"Node {node_id} local position does not match ns-3 position. "
                    f"Error: {error:.2f} m"
                )

    def send_packet(self, packet: SimPacket) -> None:
        self.bridge.send_ingress_packet(packet)

    def fetch_packets(self) -> None:
        packets = self.bridge.read_egress_packets()
        for packet in packets:
            self._validate_node_id(packet.node_id)
            self.node_packets[packet.node_id].append(packet)

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
            (self.num_gcs, NodeType.GCS),
            (self.num_drones, NodeType.DRONE),
            (self.num_users, NodeType.USER),
        ):
            prefix = self.node_type_to_prefix[node_type]
            for i in range(count):
                node = SimNode(
                    node_id=node_id,
                    node_type=node_type,
                    name=f"{node_type}{i}",
                    addr=prefix + str(i + 1),
                    position=np.zeros(3),
                )
                self.nodes.append(node)
                self.node_packets[node_id] = []
                node_id += 1

    def _launch_ns3_simulator(self, wait: float = 1.0) -> None:
        sim_cmd = (
            "scratch/swarm-net-sim/main "
            f"--nGCS={self.num_gcs} --nUAV={self.num_drones} --nUser={self.num_users}"
        )
        self.ns3_process = subprocess.Popen(
            ["./ns3", "run", sim_cmd],
            cwd=self._NS3_ROOT,
            preexec_fn=os.setsid,
        )
        time.sleep(wait)

    def _terminate_ns3_simulator(self, timeout: float = 1.0) -> None:
        if self.ns3_process and self.ns3_process.poll() is None:
            logger.info("ns-3 process is still running. Terminating...")
            process_group = os.getpgid(self.ns3_process.pid)
            os.killpg(process_group, signal.SIGTERM)
            try:
                self.ns3_process.wait(timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "ns-3 did not stop gracefully; killing its process group."
                )
                os.killpg(process_group, signal.SIGKILL)
                self.ns3_process.wait()
        self.ns3_process = None
        logger.info("ns-3 process terminated.")

    def _verify_ns3_connection(self, max_attempts: int = 2) -> None:
        logger.info("Verifying ns-3 connection...")
        is_running = False
        for _ in range(max_attempts):
            is_running = self.bridge.is_ns3_running()
            if is_running:
                break
        if not is_running:
            raise RuntimeError("ns-3 simulator is not responding.")
        logger.info("ns-3 connection verified.")

    def _verify_ns3_nodes(self) -> None:
        logger.info("Verifying ns-3 nodes...")
        addresses = self.bridge.request_node_addresses()
        self._validate_complete_node_set(addresses)
        for node_id, node_addr in addresses.items():
            self._validate_node_id(node_id)
            node = self.nodes[node_id]
            if node.node_id != node_id:
                raise RuntimeError(
                    f"ns-3 node id {node_id} does not match local node id "
                    f"{node.node_id}"
                )
            if node.addr != node_addr:
                raise RuntimeError(
                    f"ns-3 node addr {node_addr} does not match local node addr "
                    f"{node.addr}"
                )
            self._validate_node_type_address(node.node_type, node.addr)
        logger.info("ns-3 nodes verified.")

    def _validate_node_id(self, node_id: int) -> None:
        if (
            isinstance(node_id, bool)
            or not isinstance(node_id, Integral)
            or node_id < 0
        ):
            raise ValueError("Node ID must be a non-negative integer")
        if node_id >= self.num_nodes:
            raise ValueError("Node ID must be lower than the number of nodes")

    def _validate_complete_node_set(self, values: dict[int, object]) -> None:
        expected = set(range(self.num_nodes))
        if set(values) != expected:
            raise ValueError(
                f"Expected data for node IDs {sorted(expected)}, got {sorted(values)}."
            )

    def _validate_node_type_address(self, node_type: NodeType, addr: str) -> None:
        octets = addr.split(".")
        if len(octets) != 4:
            raise ValueError(f"Address must have 4 bytes but {len(octets)} were given")

        if not all(0 <= int(octet) <= 255 for octet in octets):
            raise ValueError("Address octets must be in range 0-255")

        prefix = self.node_type_to_prefix[node_type]
        if not addr.startswith(prefix):
            raise ValueError(
                f"Node of type '{node_type}' must have address in '{prefix}x' format."
                f" But {addr} was given."
            )

    def _validate_number_of_nodes(self) -> None:
        counts = {
            "GCSs": self.num_gcs,
            "drones": self.num_drones,
            "users": self.num_users,
        }
        for name, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                raise ValueError(f"Number of {name} must be a non-negative integer.")

        total = sum(counts.values())
        if total < 1:
            raise ValueError(
                "At least one node (GCS, drone, or user) must be present in the simulation."
            )
        if total > self.MAX_NODES:
            raise ValueError(
                f"The ns-3 bridge supports at most {self.MAX_NODES} nodes per "
                f"simulation; got {total}."
            )
