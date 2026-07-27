import unittest

from sim.network.network_simulator import NetworkSimulator, NodeType


class NetworkSimulatorModelTests(unittest.TestCase):
    def test_node_ids_and_types_follow_agent_registration_order(self):
        simulator = object.__new__(NetworkSimulator)
        simulator.num_gcs = 1
        simulator.num_drones = 2
        simulator.num_users = 3
        simulator.nodes = []
        simulator.node_packets = {}

        simulator._create_nodes()

        self.assertEqual([node.node_id for node in simulator.nodes], list(range(6)))
        self.assertEqual(
            [node.node_type for node in simulator.nodes],
            [
                NodeType.GCS,
                NodeType.DRONE,
                NodeType.DRONE,
                NodeType.USER,
                NodeType.USER,
                NodeType.USER,
            ],
        )

    def test_ns3_path_is_derived_from_the_package_location(self):
        self.assertTrue(str(NetworkSimulator._NS3_ROOT).endswith("ns3/ns-3"))

    def test_node_count_respects_the_bridge_datagram_limit(self):
        simulator = object.__new__(NetworkSimulator)
        simulator.num_gcs = 1
        simulator.num_drones = NetworkSimulator.MAX_NODES
        simulator.num_users = 0

        with self.assertRaisesRegex(ValueError, "at most"):
            simulator._validate_number_of_nodes()


if __name__ == "__main__":
    unittest.main()
