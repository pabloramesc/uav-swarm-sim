import math
from typing import Literal

import numpy as np

from .network_interface import NetworkInterface
from .sim_bridge import SimPacket
from .swarm_packets import PositionPacket

BroadcastMode = Literal["local", "global"]


class BroadcastService:
    def __init__(
        self,
        interface: NetworkInterface,
        interval: float,
        mode: BroadcastMode,
        jitter: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        self.iface = interface
        self.interval = float(interval)
        self.mode = mode
        self.jitter = float(jitter)
        self.rng = rng if rng is not None else np.random.default_rng()
        if not math.isfinite(self.interval) or self.interval <= 0.0:
            raise ValueError("Broadcast interval must be positive and finite.")
        if not math.isfinite(self.jitter) or self.jitter < 0.0:
            raise ValueError("Broadcast jitter must be non-negative and finite.")

        if self.mode == "global":
            self.bcast_addr = self.iface.broadcast_address
        elif self.mode == "local":
            self.bcast_addr = "255.255.255.255"
        else:
            raise ValueError("Not valid broadcast mode.")

        self.next_time: float = 0.0

    def reset(self) -> None:
        self.next_time = 0.0

    def schedule(self, now: float) -> bool:
        return now >= self.next_time

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
            delay = self.rng.normal(self.interval, self.interval * self.jitter)
            self.next_time = now + max(delay, 0)
