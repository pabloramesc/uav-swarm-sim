from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import logging

from .broadcast_service import BroadcastService
from .network_interface import NetworkInterface, SimPacket
from .network_simulator import NetworkSimulator, NodeType
from .positions_provider import PositionsProvider
from .swarm_packets import DataPacket, PacketType, parse_packet, PositionPacket

logger = logging.getLogger(__name__)


@dataclass
class SwarmMessage:
    source_id: int
    timestamp: float
    txt: str


class SwarmLink:
    def __init__(
        self,
        agent_id: int,
        network_sim: NetworkSimulator,
        local_bcast_interval: Optional[float] = None,
        global_bcast_interval: Optional[float] = None,
        position_timeout: float = 5.0,
    ):
        self.agent_id = agent_id
        self.network = network_sim
        self.iface = NetworkInterface(agent_id, network_sim)

        self.time: float = 0.0
        self.position: np.ndarray = np.zeros(3)
        self.data_packets: deque[DataPacket] = deque(maxlen=1024)
        self.send_counter: int = 0
        self.recv_counter: int = 0

        self.position_provider = PositionsProvider(
            agent_id, network_sim, position_timeout
        )

        self.broadcasters: dict[str, BroadcastService] = {}
        if local_bcast_interval is not None:
            self.broadcasters["local"] = BroadcastService(
                interface=self.iface, interval=local_bcast_interval, mode="local"
            )
        if global_bcast_interval is not None:
            self.broadcasters["global"] = BroadcastService(
                interface=self.iface, interval=global_bcast_interval, mode="global"
            )

    def update(self, time: float, position: np.ndarray) -> None:
        self.time = time
        self.position = position.copy()

        # Receive raw packets
        for raw in self.iface.receive():
            try:
                pkt = parse_packet(raw.data)
            except Exception:
                logger.error(f"Failed to parse packet: {raw}")
                continue

            if isinstance(pkt, DataPacket):
                self.data_packets.append(pkt)

            elif isinstance(pkt, PositionPacket):
                self.position_provider.process(pkt, time)

        # Prune stale positions
        self.position_provider.prune(time)

        # Broadcast as needed
        for svc in self.broadcasters.values():
            svc.update(time, self.position)

    def send_message(self, msg: str, dst_addr: str) -> None:
        pkt = DataPacket()
        pkt.set_header_fields(
            agent_id=self.agent_id,
            packet_id=self.iface.tx_packet_counter,
            timestamp=self.time,
        )
        pkt.set_payload(msg.encode())

        sim_pkt = SimPacket(
            node_id=self.iface.node_id,
            src_addr=self.iface.node_address,
            dst_addr=dst_addr,
            data=pkt.serialize(),
        )
        self.iface.send(sim_pkt)
        self.send_counter += 1

    def get_messages(self, clear: bool = False) -> list[SwarmMessage]:
        messages: list[SwarmMessage] = []
        for pkt in self.data_packets:
            msg = SwarmMessage(
                source_id=int(pkt.agent_id),
                timestamp=float(pkt.timestamp),
                txt=pkt.payload.decode(),
            )
            messages.append(msg)
            self.recv_counter += 1

        if clear:
            self.data_packets.clear()

        return messages

    def get_positions(
        self, node_type: Optional[NodeType] = None
    ) -> dict[int, np.ndarray]:
        return self.position_provider.get_positions(node_type)

    def is_connected(self, node_id: Optional[int] = None) -> bool:
        return self.position_provider.is_connected(node_id)
