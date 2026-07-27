import unittest
from collections import deque

import numpy as np

from sim.network.swarm_link import SwarmLink
from sim.network.swarm_packets import DataPacket, parse_packet


class FakeInterface:
    def __init__(self):
        self.node_id = 3
        self.node_address = "10.0.0.4"
        self.tx_packet_counter = 7
        self.rx_packet_counter = 4
        self.sent_packet = None

    def send(self, packet):
        self.sent_packet = packet
        self.tx_packet_counter += 1

    def reset(self):
        self.tx_packet_counter = 0
        self.rx_packet_counter = 0


class ResetProbe:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class SwarmLinkTests(unittest.TestCase):
    def test_send_message_returns_the_serialized_packet_id(self):
        link = object.__new__(SwarmLink)
        link.agent_id = 3
        link.time = 1.25
        link.iface = FakeInterface()
        link.send_counter = 0

        packet_id = link.send_message("hello", "10.0.0.255")

        self.assertEqual(packet_id, 7)
        self.assertEqual(link.send_counter, 1)
        self.assertEqual(link.iface.tx_packet_counter, 8)
        self.assertEqual(link.iface.sent_packet.dst_addr, "10.0.0.255")

        packet = parse_packet(link.iface.sent_packet.data)
        self.assertIsInstance(packet, DataPacket)
        self.assertEqual(int(packet.agent_id), 3)
        self.assertEqual(int(packet.packet_id), packet_id)
        self.assertEqual(packet.payload, b"hello")

    def test_reset_clears_episode_local_state(self):
        link = object.__new__(SwarmLink)
        link.time = 12.0
        link.position = np.ones(3)
        link.data_packets = deque([DataPacket()])
        link.send_counter = 8
        link.recv_counter = 9
        link.iface = FakeInterface()
        link.position_provider = ResetProbe()
        link.broadcasters = {"local": ResetProbe(), "global": ResetProbe()}

        link.reset()

        self.assertEqual(link.time, 0.0)
        np.testing.assert_array_equal(link.position, np.zeros(3))
        self.assertEqual(len(link.data_packets), 0)
        self.assertEqual(link.send_counter, 0)
        self.assertEqual(link.recv_counter, 0)
        self.assertEqual(link.iface.tx_packet_counter, 0)
        self.assertEqual(link.iface.rx_packet_counter, 0)
        self.assertEqual(link.position_provider.reset_count, 1)
        self.assertTrue(
            all(service.reset_count == 1 for service in link.broadcasters.values())
        )


if __name__ == "__main__":
    unittest.main()
