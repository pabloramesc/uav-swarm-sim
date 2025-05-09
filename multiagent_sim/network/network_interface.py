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
        network_sim: NetworkSimulator,
    ):
        self.node_id = node_id
        self.network = network_sim

        self.node_addr = self.network.get_node_address(self.node_id)

        self.tx_packet_counter = 0
        self.rx_packet_counter = 0

    def send(self, packet: SimPacket) -> None:
        """Send a packet through the network simulator."""
        self.network.send_packet(packet)
        self.tx_packet_counter += 1

    def receive(self, delete: bool = True) -> list[SimPacket]:
        """Receive packets for this node."""
        packets = self.network.get_node_packets(self.node_id, delete=delete)
        self.rx_packet_counter += len(packets)
        return packets

    def get_node_address(self) -> str:
        """Return the IPv4 address of the node."""
        return self.node_addr

    def get_broadcast_address(self) -> str:
        return self.network.get_broadcast_address()

    def __repr__(self) -> str:
        return (
            f"NetworkInterface(node_id={self.node_id}, type={self.node_type}, "
            f"type_id={self.type_id}, addr='{self.node_addr}')"
        )
