from typing import Literal
from dataclasses import dataclass
import subprocess
import time
import numpy as np

from .sim_bridge import SimBridge, SimPacket


NODE_TYPE = Literal["gcs", "uav", "user"]
NODE_TYPE_PREFIX: dict[NODE_TYPE, str] = {
    "gcs": "10.0.1.",
    "uav": "10.0.2.",
    "user": "10.0.3.",
}


@dataclass
class SimNode:
    node_id: int
    type_id: int
    type: NODE_TYPE
    addr: str = "0.0.0.0"


class NetworkSimulator:

    def __init__(self, num_gcs: int, num_uavs: int, num_users: int):
        self.num_gcs = num_gcs
        self.num_uavs = num_uavs
        self.num_users = num_users
        
        self.nodes: list[SimNode] = []
        self.node_packets: dict[int, list[SimPacket]] = {}
        self._create_nodes()

        self.bridge = SimBridge()

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def launch_simulator(self, max_attempts: int = 1, verbose: bool = True) -> None:
        for attempt in range(1, max_attempts + 1):
            try:
                if verbose:
                    print(
                        "Initializing NS-3 simulator... "
                        f"(attempt {attempt}/{max_attempts})"
                    )

                self._launch_ns3_simulator()
                if verbose:
                    print(
                        f"NS-3 simulator launched for {self.num_nodes} nodes "
                        f"({self.num_gcs} GCSs, {self.num_uavs} UAVs, and {self.num_users} users)."
                    )

                self._verify_ns3_connection()
                self._verify_ns3_nodes()

            except Exception:
                if verbose:
                    print(
                        f"Failed to launch NS-3 simulator. Retrying..."
                        if attempt < max_attempts
                        else "All attempts failed. Skipping."
                    )

    def set_node_positions(self, positions: np.ndarray) -> None:
        if positions.shape != (self.num_nodes, 3):
            raise ValueError(
                f"Positions must be a numpy array with shape ({self.num_nodes}, 3)"
            )
        node_id_pos = {id: pos for id, pos in enumerate(positions)}
        self.bridge.set_node_positions(node_id_pos)

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

    def stop_simulator(self, timeout: float = 1.0, verbose: bool = True) -> None:
        if verbose:
            print("Terminating NS-3 simulator...")

        self.bridge.stop_ns3()
        time.sleep(timeout)

        if self.ns3_process.poll() is None:  # Check if the process is still running
            if verbose:
                print("NS-3 process is still running. Terminating...")
            self.ns3_process.terminate()  # Send termination signal
            self.ns3_process.wait(timeout)  # Wait for the process to terminate

        if verbose:
            print("NS-3 process terminated.")

    def ip_address_to_node_id(self, ip_address: str) -> int:
        """
        Convert an IP address to a node ID.

        Args:
            ip_address (str): The IP address to convert.

        Returns:
            int: The corresponding node ID.

        Raises:
            ValueError: If the IP address does not match any node.
        """
        for node in self.nodes:
            if node.addr == ip_address:
                return node.node_id
        raise ValueError(f"No node found with IP address {ip_address}")

    def node_id_to_ip_address(self, node_id: int) -> str:
        """
        Convert a node ID to an IP address.

        Args:
            node_id (int): The node ID to convert.

        Returns:
            str: The corresponding IP address.

        Raises:
            ValueError: If the node ID is invalid.
        """
        self._validate_node_id(node_id)
        return self.nodes[node_id].addr

    def _create_nodes(self) -> None:
        self.nodes: list[SimNode] = []
        self.node_addresses: list[str] = []
        node_id = 0

        for id in range(self.num_gcs):
            gcs = SimNode(
                node_id=node_id, type_id=id, type="gcs", addr=f"10.0.1.{id+1}"
            )
            self.nodes.append(gcs)
            self.node_packets[node_id] = []
            node_id += 1

        for id in range(self.num_uavs):
            uav = SimNode(
                node_id=node_id, type_id=id, type="uav", addr=f"10.0.2.{id+1}"
            )
            self.nodes.append(uav)
            self.node_packets[node_id] = []
            node_id += 1

        for id in range(self.num_users):
            user = SimNode(
                node_id=node_id, type_id=id, type="user", addr=f"10.0.3.{id+1}"
            )
            self.nodes.append(user)
            self.node_packets[node_id] = []
            node_id += 1

    def _rewrite_ns3_code(self) -> None:
        subprocess.run(["sh", "update_code.sh"], cwd="./network_sim", check=True)

    def _launch_ns3_simulator(self) -> None:
        sim_cmd = (
            "scratch/swarm-net-sim/main "
            f"--nGCS={self.num_gcs} --nUAV={self.num_uavs} --nUser={self.num_users}"
        )
        self.ns3_process = subprocess.Popen(
            ["./ns3", "run", sim_cmd], cwd="./network_sim/ns-3"
        )

    def _verify_ns3_connection(self, max_attempts: int = 2) -> None:
        is_running = False
        for _ in range(max_attempts):
            is_running = self.bridge.is_ns3_running()
            if is_running:
                break

        if not is_running:
            raise Exception("Unable to connect to NS-3 simulator")

    def _verify_ns3_nodes(self) -> None:
        addresses = self.bridge.get_node_addresses()
        for id, addr in addresses.items():
            node = self.nodes[id]
            if node.node_id != id:
                raise Exception(
                    f"NS-3 node id {id} does not match local node id {node.node_id}"
                )
            if node.addr != addr:
                raise Exception(
                    f"NS-3 node addr {addr} does not match local node addr {node.addr}"
                )
            self._validate_node_type_address(node.type, node.addr)

    def _validate_node_id(self, id: int) -> None:
        if id < 0:
            raise ValueError("Node ID must be a positive integer")
        if id >= self.num_nodes:
            raise ValueError("Node ID must be lower than the number of nodes")

    def _validate_node_type_address(self, node_type: NODE_TYPE, addr: str) -> None:
        octets = addr.split(".")
        if len(octets) != 4:
            raise ValueError(f"Address must have 4 bytes but {len(octets)} were given")

        if not all(0 <= int(octet) <= 255 for octet in octets):
            raise ValueError("Address octets must be in range 0-255")

        prefix = NODE_TYPE_PREFIX[node_type]
        if not addr.startswith(prefix):
            raise ValueError(
                f"Node of type '{node_type}' must have address in '{prefix}x' format."
                f"But {addr} was given."
            )
