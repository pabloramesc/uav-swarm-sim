"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import logging
import time
from dataclasses import dataclass

from enum import IntEnum, unique
import numpy as np

from .ipc_socket import IpcMessage, IpcSocket

# Create a logger for this module
logger = logging.getLogger("SIM:SimBridge")
logger.setLevel(logging.NOTSET)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


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

    def __init__(self):
        self.sock = IpcSocket(
            addr="127.0.0.1", port=9001, ns3_port=9000, buffer_size=1000
        )
        self.sock.start_reading()

    def is_ns3_running(self, timeout: float = 1.0) -> bool:
        logger.debug("Sending DO_NOTHING command to NS-3...")

        msg = SimMessage(command=SimCommandCode.DO_NOTHING)
        self.sock.send_to_ns3(msg.to_bytes())

        time.sleep(timeout)

        ipc_msg = self.sock.get_last_message()

        if ipc_msg is None:
            logger.debug("No package received from NS-3")
            return False

        msg = self._ipc_message_to_sim_message(ipc_msg)

        if len(msg.payload) > 1:
            logger.debug("Received message is longer than 1")
            return False

        if msg.command != SimCommandCode.DO_NOTHING:
            logger.debug("Received command is not DO_NOTHING")
            return False

        logger.debug("Received heartbeat response. NS-3 is alive.")

        return True

    def set_node_positions(self, node_id_pos: dict[int, np.ndarray]) -> None:
        logger.debug("Sending SET_POSITIONS command to NS-3...")

        msg = SimMessage(command=SimCommandCode.SET_POSITIONS)

        for id, pos in node_id_pos.items():
            logger.debug(f"Processing node ID {id} with position {pos}...")

            if not isinstance(id, int) or id < 0 or id > 255:
                raise ValueError("Node id must be an integer between 0 and 255")

            pos = np.asarray(pos)
            if not isinstance(pos, np.ndarray) or pos.shape != (3,):
                raise ValueError("Node position must be a numpy array with shape (3,)")

            px, py, pz = np.float32(pos[0]), np.float32(pos[1]), np.float32(pos[2])

            msg.payload += id.to_bytes(length=1)
            msg.payload += px.tobytes()
            msg.payload += py.tobytes()
            msg.payload += pz.tobytes()

        self.sock.send_to_ns3(msg.to_bytes())

    def get_node_positions(self, timeout: float = 1.0) -> dict[int, np.ndarray]:
        logger.debug("Sending REQUEST_POSITIONS command to NS-3...")

        # Send the request command to NS-3
        request_msg = SimMessage(command=SimCommandCode.REQUEST_POSITIONS)
        self.sock.send_to_ns3(request_msg.to_bytes())

        time.sleep(timeout)

        ipc_msg = self.sock.get_last_message()

        if ipc_msg is None:
            logger.debug("No response received.")
            return {}

        reply_msg = self._ipc_message_to_sim_message(ipc_msg)

        if len(reply_msg.payload) == 0:
            logger.debug("Empty response received.")
            return {}

        if reply_msg.command != SimCommandCode.REPLY_ALL_POSITIONS:
            logger.debug(f"Unexpected reply code: {reply_msg.command}")
            return {}

        logger.debug("Processing REPLY_ALL_POSITIONS from NS-3...")

        # Parse the response
        positions = {}
        offset = 0
        while offset + 13 <= len(reply_msg.payload):
            # Each entry: 1 byte (ID) + 3 floats (4 bytes each)
            entry = reply_msg.payload[offset : offset + 13]
            node_id = entry[0]
            pos = np.frombuffer(entry[1:13], dtype=np.float32)

            logger.debug(f"Node {node_id} position: {pos}")

            positions[node_id] = pos
            offset += 13

        logger.debug(f"Final parsed positions: {positions}")
        return positions

    def get_node_addresses(self, timeout: float = 1.0) -> dict[int, str]:
        logger.debug("Sending REQUEST_ADDRESSES command to NS-3...")

        # Send the request command to NS-3
        request_msg = SimMessage(command=SimCommandCode.REQUEST_ADDRESSES)
        self.sock.send_to_ns3(request_msg.to_bytes())

        time.sleep(timeout)  # Wait for the response

        ipc_msg = self.sock.get_last_message()

        if ipc_msg is None:
            logger.debug("No response received.")
            return {}

        reply_msg = self._ipc_message_to_sim_message(ipc_msg)

        if len(reply_msg.payload) == 0:
            logger.debug("Empty response received.")
            return {}

        if reply_msg.command != SimCommandCode.REPLY_ALL_ADDRESSES:
            logger.debug(f"Unexpected reply code: {reply_msg.command}")
            return {}

        logger.debug("Processing REPLY_ALL_ADDRESSES from NS-3...")

        # Parse the response
        addresses = {}
        offset = 0
        while offset + 5 <= len(reply_msg.payload):
            # Each entry: 1 byte (node_id) + 4 bytes (IPv4 address)
            entry = reply_msg.payload[offset : offset + 5]
            node_id = entry[0]
            ip_addr = self._bytes_to_ipv4(entry[1:5])

            logger.debug(f"Node {node_id} address: {ip_addr}")

            addresses[node_id] = ip_addr
            offset += 5

        logger.debug(f"Final parsed addresses: {addresses}")
        return addresses

    def get_ns3_time(self, timeout: float = 1.0) -> float:
        logger.debug("Sending REQUEST_SIM_TIME command to NS-3...")

        request_msg = SimMessage(command=SimCommandCode.REQUEST_SIM_TIME)
        self.sock.send_to_ns3(request_msg.to_bytes())

        time.sleep(timeout)

        ipc_msg = self.sock.get_last_message()
        if ipc_msg is None:
            logger.debug("No response received for sim time.")
            return None

        reply_msg = self._ipc_message_to_sim_message(ipc_msg)

        if reply_msg.command != SimCommandCode.REPLY_SIM_TIME:
            logger.debug(f"Unexpected reply code: {reply_msg.command}")
            return None

        if len(reply_msg.payload) != 8:
            logger.debug(
                f"Invalid payload size. Expected 8 bytes, got {len(reply_msg.payload)}"
            )
            return None

        sim_time = np.frombuffer(reply_msg.payload, dtype=np.float64)[0]
        logger.debug(f"Received simulation time: {sim_time} s")
        return float(sim_time)

    def stop_ns3(self):
        logger.debug("Sending STOP_SIMULATION command to NS-3...")

        msg = SimMessage(command=SimCommandCode.STOP_SIMULATION)
        self.sock.send_to_ns3(msg.to_bytes())

        self.sock.close()

    def send_ingress_packet(self, packet: SimPacket) -> None:
        logger.debug(f"Sending INGRESS_PACKET to NS-3 for Node {packet.node_id}...")

        # Validate node_id
        self._validate_node_id(packet.node_id)

        # Validate and convert IP addresses
        src_addr_bytes = self._convert_ipv4_to_bytes(packet.src_addr)
        dst_addr_bytes = self._convert_ipv4_to_bytes(packet.dst_addr)

        # Construct the packet
        msg = SimMessage(command=SimCommandCode.INGRESS_PACKET)
        msg.payload += packet.node_id.to_bytes(length=1)
        msg.payload += src_addr_bytes
        msg.payload += dst_addr_bytes
        msg.payload += packet.data  # Append the payload

        self.sock.send_to_ns3(msg.to_bytes())

    def read_egress_packets(self) -> list[SimPacket]:
        packets: list[SimPacket] = []
        messages = self.sock.get_all_messages()

        for ipc_msg in messages:
            msg = self._ipc_message_to_sim_message(ipc_msg)

            if msg.command == SimCommandCode.EGRESS_PACKET:
                packet = self._sim_message_to_sim_packet(msg)
                packets.append(packet)
                logger.debug("Egress packet added to buffer.")

            else:
                logger.debug(f"Unexpected reply code: {msg.command}. Ignoring.")

        return packets

    def _validate_node_id(self, node_id: int) -> None:
        if not isinstance(node_id, int) or node_id < 0 or node_id > 255:
            raise ValueError("Node ID must be an integer between 0 and 255")
        return

    def _convert_ipv4_to_bytes(self, addr: str) -> bytes:
        """Validate and convert an IPv4 address string to 4-byte representation."""
        try:
            return bytes([int(octet) for octet in addr.split(".")])
        except ValueError:
            raise ValueError(
                f"Invalid IPv4 address format: {addr}. Use 'x.x.x.x' format."
            )

    def _ipc_message_to_sim_message(self, ipc_msg: IpcMessage) -> SimMessage:
        """Convert IpcMessage from the socket into a SimMessage object."""
        command = SimCommandCode(ipc_msg.data[0])
        payload = ipc_msg.data[1:]
        return SimMessage(command=command, payload=payload)

    def _sim_message_to_sim_packet(self, sim_msg: SimMessage) -> SimPacket:
        """Convert a SimMessage object into a SimPacket object."""
        if len(sim_msg.payload) < 9 + 16:
            # Minimum size: node_id (1 byte) + src_addr (4 bytes) + dst_addr (4 bytes)
            # + ingress_time (8 bytes) + egress_time (8 bytes)
            raise ValueError(
                "Invalid SimMessage payload size for SimPacket conversion."
            )

        node_id = sim_msg.payload[0]
        src_addr = self._bytes_to_ipv4(sim_msg.payload[1:5])
        dst_addr = self._bytes_to_ipv4(sim_msg.payload[5:9])
        data = sim_msg.payload[9:-16] if len(sim_msg.payload) > 9 else b""
        ingress_time = np.frombuffer(sim_msg.payload[-16:-8], dtype=np.float64)[0]
        egress_time = np.frombuffer(sim_msg.payload[-8:], dtype=np.float64)[0]

        logger.debug(
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

    def _bytes_to_ipv4(self, addr_bytes: bytes) -> str:
        """Convert 4-byte representation of an IPv4 address to a string."""
        if len(addr_bytes) != 4:
            raise ValueError("Invalid IPv4 address byte length.")
        return ".".join(map(str, addr_bytes))
