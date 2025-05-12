from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike


class PacketType(Enum):
    DATA = 0x00
    POSITION = 0x01
    ACKNOWLEDGE = 0xFF


class SwarmPacket(ABC):
    """
    Base class for all swarm packets.

    Header format (8 bytes):
    [ Packet Type (1 byte) | Agent ID (1 byte) | Packet ID (2 bytes) | Timestamp (4 bytes) ]
    """

    HEADER_SIZE = 8  # header size in bytes

    def __init__(self, packet_type: PacketType):
        self.packet_type = packet_type
        self.agent_id: np.uint8 = None
        self.packet_id: np.uint16 = None
        self.timestamp: np.float32 = None
        self.payload: bytes = b""

    def set_header_fields(
        self, agent_id: int, packet_id: int, timestamp: float
    ) -> None:
        if not 0 <= agent_id <= 255:
            raise ValueError(f"Agent ID must be between 0 and 255, got {agent_id}")

        self.agent_id = np.uint8(agent_id)
        self.packet_id = np.uint16(int(packet_id) & 0xFFFF)
        self.timestamp = np.float32(timestamp)

    def _build_header(self) -> bytes:
        header = b""
        header += self.packet_type.value.to_bytes(length=1)
        header += self.agent_id.tobytes()
        header += self.packet_id.tobytes()
        header += self.timestamp.tobytes()
        return header

    def _parse_header(self, packet: bytes) -> None:
        packet_type = PacketType(packet[0])
        if self.packet_type != packet_type:
            raise ValueError(f"Expected {self.packet_type}, got {packet_type}")

        self.agent_id = np.frombuffer(packet[1:2], dtype=np.uint8)[0]
        self.packet_id = np.frombuffer(packet[2:4], dtype=np.uint16)[0]
        self.timestamp = np.frombuffer(packet[4:8], dtype=np.float32)[0]

    def serialize(self) -> bytes:
        return self._build_header() + self.payload

    @abstractmethod
    def deserialize(self, packet: bytes) -> None:
        """Deserialize the packet from bytes."""
        pass

    def _build_description(self) -> list[str]:
        return [
            f"agent_id={self.agent_id}",
            f"packet_id={self.packet_id}",
            f"timestamp={self.timestamp:.4f}",
            f"payload={self.payload}",
        ]

    def __str__(self) -> str:
        info = self._build_description()
        return f"SwarmPacket({", ".join(info)})"

    def __repr__(self) -> str:
        return self.__str__()


class DataPacket(SwarmPacket):

    def __init__(self):
        super().__init__(PacketType.DATA)

    def set_payload(self, data: bytes) -> None:
        self.payload = data

    def deserialize(self, packet: bytes) -> None:
        if len(packet) < self.HEADER_SIZE:
            raise ValueError(
                f"DATA packet must have at least {self.HEADER_SIZE} bytes, got {len(packet)}"
            )
        self._parse_header(packet)
        self.payload = packet[self.HEADER_SIZE :]

    def __str__(self):
        info = self._build_description()
        return f"DataPacket({", ".join(info)})"

    def __repr__(self):
        return self.__str__()


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
                f"POSITION packet must be exactly {self.EXPECTED_LENGTH} bytes, got {len(packet)}"
            )
        self._parse_header(packet)
        self.payload = packet[self.HEADER_SIZE : self.EXPECTED_LENGTH]

    def __str__(self):
        info = self._build_description()[:-1]
        info.append("position=" + ", ".join([f"{p:.2f}" for p in self.get_position()]))
        return f"PositionPacket({", ".join(info)})"

    def __repr__(self):
        return self.__str__()


class AcknowledgePacket(SwarmPacket):
    def __init__(self):
        super().__init__(PacketType.ACKNOWLEDGE)

    def set_ack(self, agent_id: int, packet_id: int, timestamp: float):
        self.set_header_fields(agent_id, packet_id, timestamp)
        self.payload = b""  # No payload

    def deserialize(self, packet: bytes) -> None:
        if len(packet) != self.HEADER_SIZE:
            raise ValueError(
                f"ACK packet must be exactly {self.HEADER_SIZE} bytes, got {len(packet)}"
            )
        self._parse_header(packet)
        self.payload = b""  # No payload

    def __str__(self):
        info = self._build_description()[:-1]
        return f"AckPacket({", ".join(info)})"

    def __repr__(self):
        return self.__str__()


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
    elif packet_type == PacketType.ACKNOWLEDGE:
        packet = AcknowledgePacket()
    else:
        raise ValueError(f"Unsuported packet type: {packet_type}")

    packet.deserialize(data)
    return packet
