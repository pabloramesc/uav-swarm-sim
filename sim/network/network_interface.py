from .network_simulator import NetworkSimulator, SimPacket


class NetworkInterface:
    """
    Represents a network interface for an agent in the simulation.
    Acts as an adapter between the Agent and the NetworkSimulator.
    """

    def __init__(
        self,
        node_id: int,
        network_sim: NetworkSimulator,
    ):
        self.node_id = node_id
        self.network_simulator = network_sim

        self.node_address = self.network_simulator.get_node(self.node_id).addr
        self.broadcast_address = self.network_simulator.get_broadcast_address()

        self.tx_packet_counter = 0
        self.rx_packet_counter = 0

    def send(self, packet: SimPacket) -> None:
        """Send a packet through the network simulator."""
        self.network_simulator.send_packet(packet)
        self.tx_packet_counter += 1

    def receive(self, delete: bool = True) -> list[SimPacket]:
        """Receive packets for this node."""
        packets = self.network_simulator.get_node_packets(self.node_id, delete=delete)
        self.rx_packet_counter += len(packets)
        return packets

    def reset(self) -> None:
        self.tx_packet_counter = 0
        self.rx_packet_counter = 0

    def __repr__(self) -> str:
        node = self.network_simulator.get_node(self.node_id)
        return (
            f"NetworkInterface(node_id={self.node_id}, "
            f"type='{node.node_type.value}', addr='{self.node_address}')"
        )
