import numpy as np
from typing import Literal
from dataclasses import dataclass

from .network_simulator import NetworkSimulator, SimNode, NodeType
from .network_interface import NetworkInterface, SimPacket
from .swarm_packets import DataPacket, PositionPacket, parse_packet, PacketType
from ..utils.logger import create_logger

BroadcastMode = Literal["local", "global"]


@dataclass
class NeighborInfo:
    node: SimNode
    time: float
    position: np.ndarray
    counter: int
    is_valid: bool = True


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
        self.global_bcast_addr = self.network_interface.broadcast_address

        self.time: float = 0.0
        self.node_position: np.ndarray = None
        self.neighbors_info: dict[int, NeighborInfo] = {}
        self.data_packets: list[DataPacket] = []

        self.next_local_bcast_time = 0.0
        self.next_global_bcast_time = 0.0

        self.logger = create_logger(name=f"SwarmLink.{self.agent_id}", level="WARNING")

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

        pass

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
            src_addr=self.network_interface.node_address,
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

        node = self.network_interface.network_simulator.get_node(source_id)

        if source_id != node.node_id:
            self.logger.warning(
                f"Node ID mismatch: {source_id} != {node.node_id}. " f"Packet: {packet}"
            )
            return

        if source_id in self.neighbors_info:
            prev_info = self.neighbors_info[source_id]
            if prev_info.counter == packet.counter:
                self.logger.info(
                    f"Ignoring duplicate packet from node {source_id}. "
                    f"Packet: {packet}. Neighbor info: {prev_info}"
                )
                return

        info = NeighborInfo(
            time=packet.timestamp,
            position=packet.get_position(),
            node=node,
            counter=packet.counter,
        )

        self.neighbors_info[source_id] = info

    def get_positions(self, node_type: NodeType = None) -> dict[int, np.ndarray]:
        poisitions: dict[int, np.ndarray] = {}
        for node_id, info in self.neighbors_info.items():
            if node_type is not None and info.node.node_type != node_type:
                continue

            if self.time - info.time > self.position_timeout or info.position is None:
                if info.is_valid:
                    info.is_valid = False
                    self.logger.info(
                        f"Position timeout for node {node_id}. "
                        f"Last update at {info.time:.2f} s. Current time: {self.time:.2f} s"
                    )
                continue

            info.is_valid = True
            poisitions[node_id] = info.position

        return poisitions

    def get_data_packets(self, clear: bool = True) -> list[DataPacket]:
        packets = self.data_packets.copy()
        if clear:
            self.data_packets.clear()
        return packets
