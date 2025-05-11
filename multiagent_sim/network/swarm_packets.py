from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike


class PacketType(Enum):
    DATA = 0x00
    POSITION = 0x01


class SwarmPacket(ABC):
    """
    Base class for all swarm packets.

    Header format (8 bytes):
    [ Packet Type (1 byte) | Counter (2 bytes) | Agent ID (1 byte) | Timestamp (4 bytes) ]
    """

    HEADER_SIZE = 8  # header size in bytes

    def __init__(self, packet_type: PacketType):
        self.packet_type = packet_type
        self.agent_id: np.uint8 = None
        self.counter: np.uint16 = None
        self.timestamp: np.float32 = None
        self.payload: bytes = b""

    def set_header_fields(self, agent_id: int, counter: int, timestamp: float) -> None:
        self.agent_id = np.uint8(agent_id)
        self.counter = np.uint16(counter)
        self.timestamp = np.float32(timestamp)

    def _build_header(self) -> bytes:
        header = b""
        header += self.packet_type.value.to_bytes(length=1)
        header += self.counter.tobytes()
        header += self.agent_id.tobytes()
        header += self.timestamp.tobytes()
        return header

    def _parse_header(self, packet: bytes) -> None:
        packet_type = PacketType(packet[0])
        if self.packet_type != packet_type:
            raise ValueError(f"Expected {self.packet_type}, got {packet_type}")

        self.counter = np.frombuffer(packet[1:3], dtype=np.uint16)[0]
        self.agent_id = np.frombuffer(packet[3:4], dtype=np.uint8)[0]
        self.timestamp = np.frombuffer(packet[4:8], dtype=np.float32)[0]

    def serialize(self) -> bytes:
        return self._build_header() + self.payload

    @abstractmethod
    def deserialize(self, packet: bytes) -> None:
        """Deserialize the packet from bytes."""
        pass


class DataPacket(SwarmPacket):

    def __init__(self):
        super().__init__(PacketType.DATA)

    def set_payload(self, data: bytes) -> None:
        self.payload = data

    def deserialize(self, packet: bytes) -> None:
        if len(packet) < SwarmPacket.HEADER_SIZE:
            raise ValueError(
                f"Packet must have at least {self.HEADER_SIZE} bytes, got {len(packet)}"
            )
        self._parse_header(packet)
        self.payload = packet[self.HEADER_SIZE :]


class PositionPacket(SwarmPacket):
    PAYLOAD_SIZE = 3 * 4  # 3 float32s (px, py, pz) in bytes
    EXPECTED_LENGTH = SwarmPacket.HEADER_SIZE + PAYLOAD_SIZE

    def __init__(self):
        super().__init__(PacketType.POSITION)

    def set_position(self, position: ArrayLike) -> None:
        arr = np.asarray(position, dtype=np.float32)
        if arr.shape != (3,):
            raise ValueError("Position must be a numpy array with shape (3,)")
        self.payload = arr.tobytes()

    def get_position(self) -> np.ndarray:
        if len(self.payload) != self.PAYLOAD_SIZE:
            raise Exception(
                f"Expected payload of {PositionPacket.payload_length} bytes, got {len(self.payload)}"
            )
        return np.frombuffer(self.payload, dtype=np.float32)

    def deserialize(self, packet: bytes) -> None:
        if len(packet) != self.EXPECTED_LENGTH:
            raise ValueError(
                f"Packet must be {self.expected_length} bytes, got {len(packet)}"
            )
        self._parse_header(packet)
        self.payload = packet[self.HEADER_SIZE : self.EXPECTED_LENGTH]


def parse_packet(data: bytes) -> SwarmPacket:
    if not data:
        raise ValueError("Packet is empty")

    try:
        packet_type = PacketType(data[0])
    except ValueError:
        raise ValueError(f"Invalid packet type code: 0x{data[0]:02x} ({data[0]})")

    if packet_type == PacketType.DATA:
        packet = DataPacket()
    elif packet_type == PacketType.POSITION:
        packet = PositionPacket()
    else:
        raise ValueError(f"Unsuported packet type: {packet_type}")

    packet.deserialize(data)
    return packet
