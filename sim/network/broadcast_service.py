from typing import Literal

import numpy as np

from .network_interface import NetworkInterface
from .swarm_packets import PositionPacket
from .sim_bridge import SimPacket


BroadcastMode = Literal["local", "global"]


class BroadcastService:
    def __init__(
        self,
        interface: NetworkInterface,
        interval: float,
        mode: BroadcastMode,
        jitter: float = 0.1,
    ):
        self.iface = interface
        self.interval = interval
        self.mode = mode
        self.jitter = jitter
        
        if self.mode == "global":
            self.bcast_addr = self.iface.broadcast_address
        elif self.mode == "local":
            self.bcast_addr = "255.255.255.255"
        else:
            raise ValueError("Not valid broadcast mode.")

        self.next_time: float = 0.0

    def schedule(self, now: float) -> bool:
        return self.interval is not None and now >= self.next_time

    def broadcast_position(self, position: np.ndarray, now: float) -> None:
        pkt = PositionPacket()
        pkt.set_header_fields(
            agent_id=self.iface.node_id,
            packet_id=self.iface.tx_packet_counter,
            timestamp=now,
        )
        pkt.set_position(position)

        sim_pkt = SimPacket(
            node_id=self.iface.node_id,
            src_addr=self.iface.node_address,
            dst_addr=self.bcast_addr,
            data=pkt.serialize(),
        )
        self.iface.send(sim_pkt)

    def update(self, now: float, position: np.ndarray) -> None:
        if self.schedule(now):
            self.broadcast_position(position, now)
            # add jitter
            delay = np.random.normal(self.interval, self.interval * self.jitter)
            self.next_time = now + max(delay, 0)
