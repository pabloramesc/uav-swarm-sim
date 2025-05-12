import numpy as np
from typing import Literal
from dataclasses import dataclass
from collections import deque

from .network_simulator import NetworkSimulator, SimNode, NodeType
from .network_interface import NetworkInterface, SimPacket
from .swarm_packets import (
    DataPacket,
    PositionPacket,
    parse_packet,
    PacketType,
    MessagePacket,
    AcknowledgePacket,
)
from ..utils.logger import create_logger

BroadcastMode = Literal["local", "global"]


@dataclass
class NeighborInfo:
    node: SimNode
    time: float
    position: np.ndarray
    counter: int
    is_valid: bool = True


@dataclass
class AckEntry:
    packet_id: int
    agent_id: int
    rx_time: float


class SwarmLink:

    def __init__(
        self,
        agent_id: int,
        network_sim: NetworkSimulator,
        local_bcast_interval: float = None,
        global_bcast_interval: float = None,
        position_timeout: float = 5.0,
        ack_to_messages: bool = False,
        store_positions: bool = True,
    ):
        self.agent_id = agent_id
        self.network_simulator = network_sim
        self.network_interface = NetworkInterface(agent_id, network_sim)

        self.local_bcast_interval = local_bcast_interval
        self.global_bcast_interval = global_bcast_interval
        self.position_timeout = position_timeout
        self.ack_to_messages = ack_to_messages
        self.store_positions = store_positions

        self.local_bcast_addr = "255.255.255.255"
        self.global_bcast_addr = self.network_interface.broadcast_address

        self.time: float = 0.0
        self.node_position: np.ndarray = None
        self.neighbors_info: dict[int, NeighborInfo] = {}
        self.data_packets: deque[DataPacket] = deque(maxlen=1024)
        self.ack_registry: deque[AckEntry] = deque(maxlen=1024)

        self.next_local_bcast_time = 0.0
        self.next_global_bcast_time = 0.0

        self.logger = create_logger(name=f"SwarmLink.{self.agent_id}", level="INFO")

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
            packet_id=self.network_interface.tx_packet_counter,
            timestamp=self.time,
        )
        position_packet.set_position(position)

        if mode == "local":
            bcast_addr = self.local_bcast_addr
        elif mode == "global":
            bcast_addr = self.global_bcast_addr
        else:
            raise ValueError(f"Invalid broacast mode: {mode}")

        sim_packet = SimPacket(
            node_id=self.agent_id,
            src_addr=self.network_interface.node_address,
            dst_addr=bcast_addr,
            data=position_packet.serialize(),
        )
        self.network_interface.send(sim_packet)

    def send_data(self, data: bytes, dst_addr: str) -> None:
        data_packet = DataPacket()
        data_packet.set_header_fields(
            agent_id=self.agent_id,
            packet_id=self.network_interface.tx_packet_counter,
            timestamp=self.time,
        )
        data_packet.set_payload(data)

        sim_packet = SimPacket(
            node_id=self.agent_id,
            src_addr=self.network_interface.node_address,
            dst_addr=dst_addr,
            data=data_packet.serialize(),
        )
        self.network_interface.send(sim_packet)

    def send_message(self, message: str, dst_addr: str) -> int:
        msg_packet = MessagePacket()
        msg_packet.set_header_fields(
            agent_id=self.agent_id,
            packet_id=self.network_interface.tx_packet_counter,
            timestamp=self.time,
        )
        msg_packet.set_message(message)

        # TODO: store sent message to process ACKs

        sim_packet = SimPacket(
            node_id=self.agent_id,
            src_addr=self.network_interface.node_address,
            dst_addr=dst_addr,
            data=msg_packet.serialize(),
        )
        self.network_interface.send(sim_packet)

        self.logger.info(
            f"Sent packet {msg_packet.packet_id} to {dst_addr} "
            f"with message: {msg_packet.get_message()}"
        )

        return msg_packet.packet_id

    def get_data_packets(self, clear: bool = True) -> list[DataPacket]:
        packets = self.data_packets.copy()
        if clear:
            self.data_packets.clear()
        return packets

    def get_positions(self, node_type: NodeType = None) -> dict[int, np.ndarray]:
        poisitions: dict[int, np.ndarray] = {}
        for node_id, info in self.neighbors_info.items():
            if node_type is not None and info.node.node_type != node_type:
                continue

            if self.time - info.time > self.position_timeout or info.position is None:
                if info.is_valid:
                    info.is_valid = False
                    self.logger.debug(
                        f"Position timeout for node {node_id}. "
                        f"Last update at {info.time:.2f} s. Current time: {self.time:.2f} s"
                    )
                continue

            info.is_valid = True
            poisitions[node_id] = info.position

        return poisitions

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
                if self.store_positions:
                    self._handle_position_packet(swarm_packet)

            elif swarm_packet.packet_type == PacketType.MESSAGE:
                self.data_packets.append(swarm_packet)
                if self.ack_to_messages:
                    self._send_ack(swarm_packet)

            elif swarm_packet.packet_type == PacketType.ACKNOWLEDGE:
                self._handle_ack_packet(swarm_packet)

            else:
                raise ValueError(f"Invalid packet type: {swarm_packet.type}")

            self.logger.debug(f"Received packet: {swarm_packet}")

        return packets

    def _handle_position_packet(self, packet: PositionPacket) -> None:
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
            if prev_info.counter == packet.packet_id:
                self.logger.info(
                    f"Ignoring duplicate packet from node {source_id}. "
                    f"Packet: {packet}. Neighbor info: {prev_info}"
                )
                return

        info = NeighborInfo(
            time=packet.timestamp,
            position=packet.get_position(),
            node=node,
            counter=packet.packet_id,
        )

        self.neighbors_info[source_id] = info

    def _handle_ack_packet(self, packet: AcknowledgePacket) -> None:
        ack = AckEntry(packet.packet_id, packet.agent_id, self.time)
        self.ack_registry.append(ack)
        self.logger.info(
            f"Received ACK from agent {packet.agent_id} to packet {packet.packet_id} at {self.time:.2f} s"
        )

    def _send_ack(self, packet: MessagePacket) -> None:
        ack_packet = AcknowledgePacket()
        ack_packet.set_ack(
            agent_id=self.agent_id, packet_id=packet.packet_id, timestamp=self.time
        )

        sim_packet = SimPacket(
            node_id=self.agent_id,
            src_addr=self.network_interface.node_address,
            dst_addr=self.network_simulator.get_node(packet.agent_id).addr,
            data=ack_packet.serialize(),
        )
        self.network_interface.send(sim_packet)

        self.logger.debug(
            f"Acknowledging packet {packet.packet_id} from agent {packet.agent_id} at {self.time} s"
        )
