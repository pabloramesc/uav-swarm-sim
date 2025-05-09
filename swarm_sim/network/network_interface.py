from .network_simulator import NetworkSimulator, SimPacket, NodeType

class NetworkInterface:
    """
    Represents a network interface for an agent in the simulation.
    Acts as an adapter between the Agent and the NetworkSimulator.

    Attributes
    ----------
    node_id : int
        Unique ID assigned by the network simulator.
    type_id : int
        Index of the agent within its type group (gcs/uav/user).
    node_type : NodeType
        Type of node: "gcs", "uav", or "user".
    addr : str
        IP address assigned by the simulator.
    """

    def __init__(
        self,
        node_id: int,
        type_id: int,
        node_type: NodeType,
        addr: str,
        network_sim: NetworkSimulator,
    ):
        self.node_id = node_id
        self.type_id = type_id
        self.node_type = node_type
        self.addr = addr
        self.network = network_sim

        self._verify_node()
        
        self.tx_packet_counter = 0
        self.rx_packet_counter = 0

    def _verify_node(self):
        """Check that the node exists and matches the network simulator's registry."""
        node = self.network.nodes[self.node_id]
        if node.type != self.node_type or node.type_id != self.type_id:
            raise ValueError(
                f"NetworkSimulator node mismatch: Expected type='{self.node_type}' and type_id={self.type_id}, "
                f"but got type='{node.type}' and type_id={node.type_id}"
            )
        if node.addr != self.addr:
            raise ValueError(
                f"NetworkSimulator IP mismatch: Expected addr={self.addr}, got {node.addr}"
            )

    def send(self, packet: SimPacket) -> None:
        """Send a packet through the network simulator."""
        self.network.send_packet(packet)
        self.tx_packet_counter += 1

    def receive(self, delete: bool = True) -> list[SimPacket]:
        """Receive packets for this node."""
        packets = self.network.get_node_packets(self.node_id, delete=delete)
        self.rx_packet_counter += len(packets)
        return packets

    def get_ip(self) -> str:
        """Return the IP address of the node."""
        return self.addr

    def __repr__(self) -> str:
        return (
            f"NetworkInterface(node_id={self.node_id}, type={self.node_type}, "
            f"type_id={self.type_id}, addr='{self.addr}')"
        )
