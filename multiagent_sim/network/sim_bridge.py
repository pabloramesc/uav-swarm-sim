"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import time
from dataclasses import dataclass
from collections import deque

from enum import IntEnum, unique
import numpy as np

from .ipc_socket import IpcMessage, IpcSocket

from ..utils.logger import create_logger, LogLevel


@unique
class SimCommandCode(IntEnum):
    # simulation control
    DO_NOTHING = 0x00
    STOP_SIMULATION = 0xFF

    # node control
    SET_POSITIONS = 0x01
    INGRESS_PACKET = 0x02
    EGRESS_PACKET = 0x03

    # simulation status requests
    REQUEST_POSITIONS = 0xA1
    REQUEST_ADDRESSES = 0xA2
    REQUEST_SIM_TIME = 0xA3

    # replies
    REPLY_ALL_POSITIONS = 0xB1
    REPLY_ALL_ADDRESSES = 0xB2
    REPLY_SIM_TIME = 0xB3


@dataclass
class SimMessage:
    command: SimCommandCode
    payload: bytes = b""

    def to_bytes(self) -> bytes:
        return bytes([self.command]) + self.payload


@dataclass
class SimPacket:
    node_id: int
    src_addr: str
    dst_addr: str
    data: bytes = b""
    ingress_time: float = None
    egress_time: float = None


class SimBridge:

    def __init__(self, loglevel: LogLevel = "INFO"):
        self.sock = IpcSocket(addr="127.0.0.1", port=9001, ns3_port=9000, max_size=1000)
        self.sock.start_reading()

        # Attributes to store last received values
        self.last_positions: dict[int, np.ndarray] = {}
        self.last_addresses: dict[int, str] = {}
        self.last_sim_time: float = None
        self.packets_rtt: deque[float] = deque(maxlen=100)

        self.packets: deque[SimPacket] = deque(maxlen=1024)
        self.replies: deque[SimMessage] = deque(maxlen=16)

        self.logger = create_logger(name="SimBridge", level=loglevel)

    @property
    def mean_rtt(self) -> float:
        if len(self.packets_rtt) > 0:
            return sum(self.packets_rtt) / len(self.packets_rtt)
        return 0.0

    def is_ns3_running(self, timeout: float = 1.0) -> bool:
        """Send heartbeat and expect DO_NOTHING reply."""
        self.logger.debug("Checking NS-3 heartbeat...")
        try:
            reply = self._send_and_receive(
                SimCommandCode.DO_NOTHING,
                expected_reply=SimCommandCode.DO_NOTHING,
                timeout=timeout,
            )

            self.logger.debug("NS-3 is alive.")
            return True

        except TimeoutError:
            self.logger.debug("No heartbeat response from NS-3.")
            return False

    def set_node_positions(self, node_positions: dict[int, np.ndarray]) -> None:
        """Send new positions for multiple nodes."""
        if node_positions is None or len(node_positions) == 0:
            raise ValueError("Node positions cannot be empty.")
        
        self.logger.debug("Setting node positions...")

        msg = SimMessage(command=SimCommandCode.SET_POSITIONS)
        for node_id, pos in node_positions.items():
            self._validate_node_id(node_id)
            pos_arr = np.asarray(pos, dtype=np.float32)
            if pos_arr.shape != (3,):
                raise ValueError("Position must be a 3-element array.")

            msg.payload += node_id.to_bytes(length=1)
            msg.payload += pos_arr.tobytes()
            self.logger.debug(f"Queued position {pos_arr} for node {node_id}.")

        self.sock.send_to_ns3(msg.to_bytes())

    def request_node_positions(self, timeout: float = 1.0) -> dict[int, np.ndarray]:
        """Request current node positions and store in last_positions."""
        self.logger.debug("Requesting node positions from NS-3...")
        reply = self._send_and_receive(
            SimCommandCode.REQUEST_POSITIONS,
            expected_reply=SimCommandCode.REPLY_ALL_POSITIONS,
            timeout=timeout,
        )

        positions = {}
        for node_id, pos in self._parse_positions(reply.payload).items():
            positions[node_id] = pos

        self.last_positions = positions
        return positions

    def request_node_addresses(self, timeout: float = 1.0) -> dict[int, str]:
        """Request current node addresses and store in last_addresses."""
        self.logger.debug("Requesting node addresses from NS-3...")
        reply = self._send_and_receive(
            SimCommandCode.REQUEST_ADDRESSES,
            expected_reply=SimCommandCode.REPLY_ALL_ADDRESSES,
            timeout=timeout,
        )
        addresses = self._parse_addresses(reply.payload)
        self.last_addresses = addresses
        return addresses

    def request_sim_time(self, timeout: float = 1.0) -> float:
        """Request current simulation time and store in last_sim_time."""
        self.logger.debug("Requesting simulation time from NS-3...")
        reply = self._send_and_receive(
            SimCommandCode.REQUEST_SIM_TIME,
            expected_reply=SimCommandCode.REPLY_SIM_TIME,
            timeout=timeout,
        )
        sim_time = float(np.frombuffer(reply.payload, dtype=np.float64)[0])
        self.last_sim_time = sim_time
        self.logger.debug(f"Received simulation time: {sim_time}.")
        return sim_time

    def send_ingress_packet(self, packet: SimPacket) -> None:
        """Send an ingress packet to the specified node."""
        self.logger.debug(f"Sending ingress packet to node {packet.node_id}...")

        # Validate node_id
        self._validate_node_id(packet.node_id)

        # Validate and convert IP addresses
        src_addr_bytes = self._ipv4_str_to_bytes(packet.src_addr)
        dst_addr_bytes = self._ipv4_str_to_bytes(packet.dst_addr)

        # Construct the packet
        msg = SimMessage(command=SimCommandCode.INGRESS_PACKET)
        msg.payload += packet.node_id.to_bytes(length=1)
        msg.payload += src_addr_bytes
        msg.payload += dst_addr_bytes
        msg.payload += packet.data  # Append the payload

        self.sock.send_to_ns3(msg.to_bytes())

    def read_egress_packets(self, clear: bool = True) -> list[SimPacket]:
        self._process_incoming_messages()
        packets = self.packets.copy()
        if clear:
            self.packets.clear()
        return packets

    def stop_simulation(self) -> None:
        """Stop the NS-3 simulation cleanly and close socket."""
        self.logger.debug("Stopping NS-3 simulation...")
        msg = SimMessage(command=SimCommandCode.STOP_SIMULATION)
        self.sock.send_to_ns3(msg.to_bytes())
        self.sock.close()

    def _process_incoming_messages(self):
        """Drain the socket and fill packet and reply buffers."""
        raw = self.sock.get_all_messages()
        for ipc in raw:
            sim = self._ipc_to_sim(ipc)

            if sim.command == SimCommandCode.EGRESS_PACKET:
                pkt = self._sim_to_packet(sim)
                self.packets.append(pkt)
                self.last_sim_time = pkt.egress_time
                self.packets_rtt.append(pkt.egress_time - pkt.ingress_time)
                self.logger.debug(f"Egress packet added to buffer: {pkt}")

            else:
                self.replies.append(sim)
                self.logger.debug(f"Sim message added to buffer: {sim}")

    def _send_and_receive(
        self,
        request: SimCommandCode,
        expected_reply: SimCommandCode,
        timeout: float,
    ) -> SimMessage:
        """Generic request/receive pattern with validation."""
        self.sock.send_to_ns3(SimMessage(command=request).to_bytes())
        deadline = time.time() + timeout

        while time.time() < deadline:
            self._process_incoming_messages()

            for idx, sim_msg in enumerate(self.replies):
                if sim_msg and sim_msg.command == expected_reply:
                    self.replies[idx] = None
                    return sim_msg

            time.sleep(1e-3)

        raise TimeoutError(f"Did not receive {expected_reply} in {timeout}s")

    def _parse_positions(self, payload: bytes) -> dict[int, np.ndarray]:
        """Convert raw payload to node positions mapping."""
        positions = {}
        offset = 0
        entry_size = 1 + 3 * 4  # id + 3 floats
        while offset + entry_size <= len(payload):
            chunk = payload[offset : offset + entry_size]
            node_id = chunk[0]
            pos = np.frombuffer(chunk[1:], dtype=np.float32)
            positions[node_id] = pos
            offset += entry_size
        return positions

    def _parse_addresses(self, payload: bytes) -> dict[int, str]:
        """Convert raw payload to node addresses mapping."""
        addresses = {}
        offset = 0
        entry_size = 1 + 4
        while offset + entry_size <= len(payload):
            chunk = payload[offset : offset + entry_size]
            node_id = chunk[0]
            addr = self._ipv4_bytes_to_str(chunk[1:])
            addresses[node_id] = addr
            offset += entry_size
        return addresses

    def _ipc_to_sim(self, ipc_msg: IpcMessage) -> SimMessage:
        """Convert IpcMessage from the socket into a SimMessage object."""
        command = SimCommandCode(ipc_msg.data[0])
        payload = ipc_msg.data[1:]
        return SimMessage(command=command, payload=payload)

    def _sim_to_packet(self, sim_msg: SimMessage) -> SimPacket:
        """Convert a SimMessage object into a SimPacket object."""
        # Footer: ingress_time (8 bytes) + egress_time (8 bytes)
        data_end = len(sim_msg.payload) - 16
        # Header: node_id (1 byte) + src_addr (4 bytes) + dst_addr (4 bytes)
        if data_end < 9:
            raise ValueError(
                "Invalid SimMessage payload size for SimPacket conversion."
            )

        node_id = sim_msg.payload[0]
        src_addr = self._ipv4_bytes_to_str(sim_msg.payload[1:5])
        dst_addr = self._ipv4_bytes_to_str(sim_msg.payload[5:9])
        data = sim_msg.payload[9:data_end] if data_end > 9 else b""
        ingress_time = np.frombuffer(
            sim_msg.payload[data_end : data_end + 8], dtype=np.float64
        )[0]
        egress_time = np.frombuffer(
            sim_msg.payload[data_end + 8 : data_end + 16], dtype=np.float64
        )[0]

        self.logger.debug(
            f"Converting SimMessage to SimPacket: node_id={node_id}, "
            f"src_addr={src_addr}, dst_addr={dst_addr}, data_size={len(data)}, data={data}, "
            f"ingress_time={ingress_time}, egress_time={egress_time}"
        )

        return SimPacket(
            node_id=node_id,
            src_addr=src_addr,
            dst_addr=dst_addr,
            data=data,
            ingress_time=ingress_time,
            egress_time=egress_time,
        )

    def _validate_node_id(self, node_id: int) -> None:
        if not isinstance(node_id, int) or node_id < 0 or node_id > 255:
            raise ValueError("Node ID must be an integer between 0 and 255")
        return

    def _ipv4_str_to_bytes(self, addr: str) -> bytes:
        """Validate and convert an IPv4 address string to 4-byte representation."""
        try:
            return bytes([int(octet) for octet in addr.split(".")])
        except ValueError:
            raise ValueError(
                f"Invalid IPv4 address format: {addr}. Use 'x.x.x.x' format."
            )

    def _ipv4_bytes_to_str(self, addr_bytes: bytes) -> str:
        """Convert 4-byte representation of an IPv4 address to a string."""
        if len(addr_bytes) != 4:
            raise ValueError("Invalid IPv4 address byte length.")
        return ".".join(map(str, addr_bytes))
