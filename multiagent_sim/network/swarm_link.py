import numpy as np
from typing import Literal
from dataclasses import dataclass

from .network_simulator import NetworkSimulator
from .network_interface import NetworkInterface, SimPacket
from .swarm_packets import DataPacket, PositionPacket, parse_packet, PacketType

BroadcastMode = Literal["local", "global"]


@dataclass
class NeighborPosition:
    time: float
    position: np.ndarray


class SwarmLink:

    def __init__(
        self,
        agent_id: int,
        network_sim: NetworkSimulator,
        local_bcast_interval: float = None,
        global_bcast_interval: float = None,
        position_timeout: float = 5.0,
    ):
        self.agent_id = agent_id
        self.network_interface = NetworkInterface(agent_id, network_sim)

        self.local_bcast_interval = local_bcast_interval
        self.global_bcast_interval = global_bcast_interval
        self.position_timeout = position_timeout

        self.local_bcast_addr = "255.255.255.255"
        self.global_bcast_addr = self.network_interface.get_broadcast_address()

        self.time: float = 0.0
        self.node_position: np.ndarray = None
        self.drone_positions: dict[int, NeighborPosition] = {}
        self.user_positions: dict[int, NeighborPosition] = {}
        self.data_packets: list[DataPacket] = []

        self.next_local_bcast_time = 0.0
        self.next_global_bcast_time = 0.0

    def update(self, time: float, position: np.ndarray) -> None:
        self.time = time
        self.position = position.copy()

        self.receive()

        if self.local_bcast_interval and self.time > self.next_local_bcast_time:
            self.broadcast_position(self.position, mode="local")
            self.next_local_bcast_time = self.time + np.random.normal(
                self.local_bcast_interval, self.local_bcast_interval / 10
            )

        if self.global_bcast_interval and self.time > self.next_global_bcast_time:
            self.broadcast_position(self.position, mode="global")
            self.next_global_bcast_time = self.time + np.random.normal(
                self.global_bcast_interval, self.global_bcast_interval / 10
            )

    def broadcast_position(self, position: np.ndarray, mode: BroadcastMode) -> None:
        position_packet = PositionPacket()
        position_packet.set_header_fields(
            agent_id=self.agent_id,
            counter=self.network_interface.tx_packet_counter,
            timestamp=self.time,
        )
        position_packet.set_position(position)

        if mode == "local":
            bcast_addr = self.local_bcast_addr
        elif mode == "global":
            bcast_addr = self.global_bcast_addr
        else:
            raise ValueError(f"Invalid broacast mode: {mode}")

        broadcast_packet = SimPacket(
            node_id=self.agent_id,
            src_addr=self.network_interface.node_addr,
            dst_addr=bcast_addr,
            data=position_packet.serialize(),
        )

        self.network_interface.send(broadcast_packet)

    def receive(self, delete: bool = True) -> list[SimPacket]:
        packets = self.network_interface.receive(delete)

        for packet in packets:
            try:
                swarm_packet = parse_packet(packet.data)
            except Exception:
                print(f"Failed to parse packet: {packet}")

            if swarm_packet.packet_type == PacketType.DATA:
                self.data_packets.append(swarm_packet)
            elif swarm_packet.packet_type == PacketType.POSITION:
                self.handle_position_packet(swarm_packet)
            else:
                raise ValueError(f"Invalid packet type: {swarm_packet.type}")

        return packets

    def handle_position_packet(self, packet: PositionPacket) -> None:
        source_id = packet.agent_id
        if source_id == self.agent_id:
            return  # ignore own packets
        agent_type, type_id = self.network_interface.network_simulator.get_node_type_id(
            source_id
        )
        pos = NeighborPosition(
            time=packet.timestamp, position=packet.get_position()
        )
        if agent_type == "user":
            self.user_positions[type_id] = pos
        elif agent_type == "uav":
            self.drone_positions[type_id] = pos
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

    def get_drone_positions(self) -> dict[int, np.ndarray]:
        drone_positions: dict[int, np.ndarray] = {}

        for node_id, pos in self.drone_positions.items():
            if pos is None:
                continue

            if self.time - pos.time < self.position_timeout:
                drone_positions[node_id] = pos.position
            else:
                continue

        return drone_positions

    def get_user_positions(self) -> dict[int, np.ndarray]:
        user_positions: dict[int, np.ndarray] = {}

        for node_id, pos in self.user_positions.items():
            if pos is None:
                continue

            if self.time - pos.time < self.position_timeout:
                user_positions[node_id] = pos.position
                continue

        return user_positions

    def get_data_packets(self, clear: bool = True) -> list[DataPacket]:
        packets = self.data_packets.copy()
        if clear:
            self.data_packets.clear()
        return packets
