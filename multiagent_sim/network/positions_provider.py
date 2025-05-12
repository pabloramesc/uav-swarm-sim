import numpy as np
from dataclasses import dataclass

from .network_simulator import NetworkSimulator, SimNode, NodeType
from .swarm_packets import PositionPacket


@dataclass
class NeighborInfo:
    node: SimNode
    last_seen: float
    position: np.ndarray
    valid: bool = True


class PositionsProvider:
    """
    Collects and prunes position broadcasts from peers.
    """

    def __init__(self, agent_id: int, network: NetworkSimulator, timeout: float):
        self.agent_id = agent_id
        self.network = network
        self.timeout = timeout
        self.neighbors: dict[int, NeighborInfo] = {}

    def process(self, packet: PositionPacket, now: float) -> None:
        """
        Handle an incoming PositionPacket, updating neighbor info.
        """
        source_id = packet.agent_id
        if source_id == self.agent_id:
            return
        node = self.network.get_node(source_id)
        info = NeighborInfo(
            node=node, last_seen=now, position=packet.get_position(), valid=True
        )
        self.neighbors[source_id] = info

    def prune(self, now: float) -> None:
        """
        Invalidate any neighbor whose last update exceeds the timeout.
        """
        for info in self.neighbors.values():
            if info.valid and now - info.last_seen > self.timeout:
                info.valid = False

    def get_positions(self, node_type: NodeType = None) -> dict[int, np.ndarray]:
        """
        Return a dict of {node_id: position} for all valid neighbors,
        optionally filtered by node type.
        """
        result: dict[int, np.ndarray] = {}
        for node_id, info in self.neighbors.items():
            if not info.valid:
                continue
            if node_type is not None and info.node.node_type != node_type:
                continue
            result[node_id] = info.position
        return result

    def is_connected(self, node_id: int = None) -> bool:
        """
        Simple connectivity check: any valid neighbor exists.
        """
        if node_id is None:
            return any(info.valid for info in self.neighbors.values())
        return self.neighbors[node_id].valid
