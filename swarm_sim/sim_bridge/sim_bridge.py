from .ipc_socket import IpcSocket, IpcMessage
import time
import numpy as np
import logging

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


class SimBridge:

    def __init__(self):
        self.sock = IpcSocket(addr="127.0.0.1", port=9001, ns3_port=9000)

    def is_ns3_running(self) -> bool:
        logging.debug("Sending CMD_DO_NOTHING to NS-3...")

        data = bytes([CMD_DO_NOTHING])
        self.sock.send_to_ns3(data)

        time.sleep(1.0)

        msg = self.sock.read_socket()

        if msg is None:
            logging.debug("No package received from NS-3")
            return False

        if len(msg.data) > 1:
            logging.debug("Received message is longer than 1")
            return False

        if msg.data[0] != CMD_DO_NOTHING:
            logging.debug("Received command is not CMD_DO_NOTHING")
            return False

        return True

    def set_node_positions(self, node_id_pos: dict[int, np.ndarray]) -> None:
        logging.debug("Sending CMD_SET_POSITIONS to NS-3...")

        data = bytes([CMD_SET_POSITIONS])

        for id, pos in node_id_pos.items():
            logging.debug(f"Processing node ID {id} with position {pos}...")

            if not isinstance(id, int) or id < 0 or id > 255:
                raise ValueError("Node id must be an integer between 0 and 255")

            pos = np.asarray(pos)
            if not isinstance(pos, np.ndarray) or pos.shape != (3,):
                raise ValueError("Node position must be a numpy array with shape (3,)")

            px, py, pz = np.float32(pos[0]), np.float32(pos[1]), np.float32(pos[2])

            data += id.to_bytes(length=1)
            data += px.tobytes()
            data += py.tobytes()
            data += pz.tobytes()

        self.sock.send_to_ns3(data)

    def get_node_positions(self) -> dict[int, np.ndarray]:
        logging.debug("Sending CMD_REQUEST_POSITIONS to NS-3...")

        # Send the request command to NS-3
        data = bytes([CMD_REQUEST_POSITIONS])
        self.sock.send_to_ns3(data)

        time.sleep(1.0)

        # Read the response from NS-3
        msg = self.sock.read_socket()

        if msg is None or len(msg.data) < 1:
            logging.debug("No response or empty response received from NS-3.")
            return {}

        if msg.data[0] != REPLY_ALL_POSITIONS:
            logging.debug(f"Unexpected reply code: {msg.data[0]}")
            return {}

        logging.debug("Processing REPLY_ALL_POSITIONS from NS-3...")

        # Parse the response
        positions = {}
        offset = 1  # Skip the reply code
        while offset + 13 <= len(
            msg.data
        ):  # Each entry: 1 byte (ID) + 3 floats (4 bytes each)
            node_id = msg.data[offset]
            offset += 1
            pos = np.frombuffer(msg.data[offset : offset + 12], dtype=np.float32)
            offset += 12
            positions[node_id] = pos
            logging.debug(f"Node {node_id} position: {pos}")

        logging.debug(f"Final parsed positions: {positions}")
        return positions

    def stop(self):
        logging.debug("Sending CMD_STOP_SIMULATION to NS-3...")

        data = bytes([CMD_STOP_SIMULATION])
        self.sock.send_to_ns3(data)

        self.sock.close()

    def send_packet(
        self, node_id: int, src_addr: str, dest_addr: str, data: bytes
    ) -> None:
        logging.debug(f"Sending CMD_INGRESS_PACKET to NS-3 for Node {node_id}...")

        # Validate node_id
        self._validate_node_id(node_id)

        # Validate and convert IP addresses
        src_addr_bytes = self._convert_ipv4_to_bytes(src_addr)
        dest_addr_bytes = self._convert_ipv4_to_bytes(dest_addr)

        # Construct the packet
        packet = bytes([CMD_INGRESS_PACKET])  # 1 byte for command
        packet += node_id.to_bytes(length=1)
        packet += src_addr_bytes
        packet += dest_addr_bytes
        packet += data  # Append the payload

        self.sock.send_to_ns3(packet)

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
