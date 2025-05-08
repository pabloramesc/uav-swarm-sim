import logging
import time
from collections import deque

from dataclasses import dataclass

import numpy as np

from .ipc_socket import IpcMessage, IpcSocket

logging.basicConfig(
    level=logging.DEBUG,
    format="[SwarmSim:SimBridge] %(levelname)s: %(message)s",
)

CMD_DO_NOTHING = 0x00
CMD_SET_POSITIONS = 0x01
CMD_REQUEST_POSITIONS = 0x02
CMD_INGRESS_PACKET = 0x03
CMD_STOP_SIMULATION = 0xFF
REPLY_ALL_POSITIONS = 0xA1
REPLY_EGRESS_PACKET = 0xA2


@dataclass
class SimMessage:
    command: int
    payload: bytes = b""

    def to_bytes(self) -> bytes:
        return bytes([self.command]) + self.payload


@dataclass
class SimPacket:
    node_id: int
    src_addr: str
    dst_addr: str
    data: bytes = b""


class SimBridge:

    def __init__(self):
        self.sock = IpcSocket(addr="127.0.0.1", port=9001, ns3_port=9000)
        self.packets: list[SimPacket] = []

    def is_ns3_running(self) -> bool:
        logging.debug("Sending CMD_DO_NOTHING to NS-3...")

        data = bytes([CMD_DO_NOTHING])
        self.sock.send_to_ns3(data)

        time.sleep(1.0)

        ipc_msg = self.sock.read_socket()

        if ipc_msg is None:
            logging.debug("No package received from NS-3")
            return False

        msg = self._ipc_message_to_sim_message(ipc_msg)

        if len(msg.payload) > 1:
            logging.debug("Received message is longer than 1")
            return False

        if msg.command != CMD_DO_NOTHING:
            logging.debug("Received command is not CMD_DO_NOTHING")
            return False

        logging.debug("Received heartbeat response. NS-3 is alive.")

        return True

    def set_node_positions(self, node_id_pos: dict[int, np.ndarray]) -> None:
        logging.debug("Sending CMD_SET_POSITIONS to NS-3...")

        msg = SimMessage(command=CMD_SET_POSITIONS)

        for id, pos in node_id_pos.items():
            logging.debug(f"Processing node ID {id} with position {pos}...")

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

    def get_node_positions(self) -> dict[int, np.ndarray]:
        logging.debug("Sending CMD_REQUEST_POSITIONS to NS-3...")

        # Send the request command to NS-3
        msg = SimMessage(command=CMD_REQUEST_POSITIONS)
        self.sock.send_to_ns3(msg.to_bytes())

        time.sleep(1.0)

        ipc_msg = self.sock.read_socket()

        if ipc_msg is None:
            logging.debug("No response received.")
            return {}

        msg = self._ipc_message_to_sim_message(ipc_msg)

        if len(msg.payload) == 0:
            logging.debug("Empty response received.")
            return {}

        if msg.command != REPLY_ALL_POSITIONS:
            logging.debug(f"Unexpected reply code: {msg.data[0]}")
            return {}

        logging.debug("Processing REPLY_ALL_POSITIONS from NS-3...")

        # Parse the response
        positions = {}
        offset = 0
        while offset + 13 <= len(msg.payload):
            # Each entry: 1 byte (ID) + 3 floats (4 bytes each)
            entry = msg.payload[offset : offset + 13]
            node_id = entry[0]
            pos = np.frombuffer(entry[1:13], dtype=np.float32)

            logging.debug(f"Node {node_id} position: {pos}")

            positions[node_id] = pos
            offset += 13

        logging.debug(f"Final parsed positions: {positions}")
        return positions

    def stop(self):
        logging.debug("Sending CMD_STOP_SIMULATION to NS-3...")

        msg = SimMessage(command=CMD_STOP_SIMULATION)
        self.sock.send_to_ns3(msg.to_bytes())

        self.sock.close()

    def send_packet(
        self, node_id: int, src_addr: str, dst_addr: str, data: bytes
    ) -> None:
        logging.debug(f"Sending CMD_INGRESS_PACKET to NS-3 for Node {node_id}...")

        # Validate node_id
        self._validate_node_id(node_id)

        # Validate and convert IP addresses
        src_addr_bytes = self._convert_ipv4_to_bytes(src_addr)
        dst_addr_bytes = self._convert_ipv4_to_bytes(dst_addr)

        # Construct the packet
        msg = SimMessage(command=CMD_INGRESS_PACKET)
        msg.payload += node_id.to_bytes(length=1)
        msg.payload += src_addr_bytes
        msg.payload += dst_addr_bytes
        msg.payload += data  # Append the payload

        self.sock.send_to_ns3(msg.to_bytes())

    def update_egress_packets(self):
        while True:
            ipc_msg = self.sock.read_socket()
            if ipc_msg is None:
                logging.debug("No messages pending in socket.")
                break

            msg = self._ipc_message_to_sim_message(ipc_msg)
            if msg.command == REPLY_EGRESS_PACKET:
                packet = self._sim_message_to_sim_packet(msg)
                self.packets.append(packet)
                logging.debug("Egress packet added to buffer.")
            else:
                logging.debug(f"Unexpected reply code: {msg.command}. Ignoring.")

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
        command = ipc_msg.data[0]
        payload = ipc_msg.data[1:]
        return SimMessage(command=command, payload=payload)

    def _sim_message_to_sim_packet(self, sim_msg: SimMessage) -> SimPacket:
        """Convert a SimMessage object into a SimPacket object."""
        if len(sim_msg.payload) < 9:
            # Minimum size: 1 byte (node_id) + 4 bytes (src_addr) + 4 bytes (dst_addr) + data
            raise ValueError(
                "Invalid SimMessage payload size for SimPacket conversion."
            )

        node_id = sim_msg.payload[0]
        src_addr = self._bytes_to_ipv4(sim_msg.payload[1:5])
        dst_addr = self._bytes_to_ipv4(sim_msg.payload[5:9])
        data = sim_msg.payload[9:] if len(sim_msg.payload) > 9 else b""

        logging.debug(
            f"Converting SimMessage to SimPacket: node_id={node_id}, "
            f"src_addr={src_addr}, dst_addr={dst_addr}, data_size={len(data)}, data={data}"
        )

        return SimPacket(
            node_id=node_id, src_addr=src_addr, dst_addr=dst_addr, data=data
        )

    def _bytes_to_ipv4(self, addr_bytes: bytes) -> str:
        """Convert 4-byte representation of an IPv4 address to a string."""
        if len(addr_bytes) != 4:
            raise ValueError("Invalid IPv4 address byte length.")
        return ".".join(map(str, addr_bytes))
