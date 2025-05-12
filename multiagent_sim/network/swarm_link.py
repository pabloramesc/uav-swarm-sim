import numpy as np

from collections import deque
from dataclasses import dataclass

from .network_simulator import NetworkSimulator, NodeType
from .network_interface import NetworkInterface, SimPacket
from .swarm_packets import parse_packet, PacketType, DataPacket
from .positions_provider import PositionsProvider
from .broadcast_service import BroadcastService


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
        local_bcast_interval: float = None,
        global_bcast_interval: float = None,
        position_timeout: float = 5.0,
    ):
        self.agent_id = agent_id
        self.network = network_sim
        self.iface = NetworkInterface(agent_id, network_sim)

        self.time: float = 0.0
        self.position: np.ndarray = np.zeros(3)
        self.data_packets: list[DataPacket] = deque(maxlen=1024)

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

    def update(self, now: float, position: np.ndarray) -> None:
        self.time = now
        self.position = position.copy()

        # Receive raw packets
        for raw in self.iface.receive():
            try:
                pkt = parse_packet(raw.data)
            except Exception:
                self.logger.warning(f"Failed to parse packet: {raw}")
                continue

            if pkt.packet_type == PacketType.DATA:
                self.data_packets.append(pkt)
            elif pkt.packet_type == PacketType.POSITION:
                self.position_provider.process(pkt, now)

        # Prune stale positions
        self.position_provider.prune(now)

        # Broadcast as needed
        for svc in self.broadcasters.values():
            svc.update(now, self.position)

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

    def get_messages(self, clear: bool = False) -> list[SwarmMessage]:
        messages: list[SwarmMessage] = []
        for pkt in self.data_packets:
            msg = SwarmMessage(pkt.agent_id, pkt.timestamp, pkt.payload.decode())
            messages.append(msg)

        if clear:
            self.data_packets.clear()

        return messages

    def get_positions(self, node_type: NodeType = None) -> dict[int, np.ndarray]:
        return self.position_provider.get_positions(node_type)

    def is_connected(self, node_id: int = None) -> bool:
        return self.position_provider.is_connected(node_id)
