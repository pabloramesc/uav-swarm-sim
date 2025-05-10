from enum import Enum
import time
import numpy as np
from numpy.typing import ArrayLike


class PacketType(Enum):
    DATA = 0x00
    POSITION = 0xA1


class DataPacket:
    min_length = 8  # header: packet type (1 byte) + agent ID (1 byte) + counter (2 bytes) + timestamp (4 bytes)

    def __init__(self):
        self.type = PacketType.DATA
        self.agent_id: np.uint8 = None
        self.counter: np.uint16 = None
        self.timestamp: np.float32 = None
        self.payload: bytes = b""

    def set_header_fields(self, agent_id: int, counter: int, timestamp: float) -> None:
        self.agent_id = np.uint8(agent_id)
        self.counter = np.uint16(counter)
        self.timestamp = np.float32(timestamp)

    def set_payload(self, data: bytes) -> None:
        self.payload = data

    def get_header(self) -> bytes:
        header = b""
        header += self.type.value.to_bytes(length=1)
        header += self.counter.tobytes()
        header += self.agent_id.tobytes()
        header += self.timestamp.tobytes()
        return header

    def serialize(self, data: bytes = None) -> bytes:
        header = self.get_header()
        payload = data if data is not None else self.payload
        return header + payload

    def deserialize(self, packet: bytes) -> None:
        if len(packet) < 9:
            raise ValueError("Packet length must be at least 9 bytes.")

        self.type = PacketType(packet[0])
        self.agent_id = np.frombuffer(packet[1:5], dtype=np.uint32)[0]
        self.timestamp = np.frombuffer(packet[5:9], dtype=np.float32)[0]
        self.payload = packet[9:]

        expected_length = 9 + len(self.payload)
        if len(packet) != expected_length:
            raise ValueError(
                f"Packet length mismatch. Expected {expected_length} bytes, got {len(packet)}"
            )


class PositionPacket(DataPacket):
    payload_length = 3 * 4  # positions [px, py, pz] in float32 (3x4 bytes)
    expected_length = (
        DataPacket.min_length + payload_length
    )  # header (9 bytes) + payload (12 bytes)

    def __init__(self):
        super().__init__()
        self.packet_type = PacketType.POSITION

    def set_position(self, position: ArrayLike) -> None:
        position = np.asarray(position, dtype=np.float32)
        if position.shape != (3,):
            raise ValueError("Position must be a numpy array with shape (3,)")
        data = position.tobytes()
        self.set_payload(data)

    def get_position(self) -> np.ndarray:
        if len(self.payload) != PositionPacket.payload_length:
            raise Exception(
                f"Payload must have {PositionPacket.payload_length} bytes, got {len(self.payload)}"
            )
        position = np.frombuffer(self.payload, dtype=np.float32)
        return position

    def deserialize(self, packet):
        if len(packet) != PositionPacket.expected_length:
            raise ValueError(
                f"Packet must have {self.expected_length} bytes, got {len(packet)}"
            )
        return super().deserialize(packet)


def parse_packet(data: bytes) -> DataPacket:
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
