from multiagent_sim.network.network_simulator import NetworkSimulator, SimPacket
import time
import numpy as np

# Configuration
NUM_UAVS = 10
SPACING = 50.0
TX_INTERVAL = 0.01  # seconds


def setup_network():
    """Initializes and configures the network simulator with node positions."""
    net_sim = NetworkSimulator(num_gcs=1, num_uavs=NUM_UAVS, num_users=2)
    net_sim.launch_simulator(max_attempts=2)

    positions = np.zeros((1 + NUM_UAVS + 2, 3))  # GCS + UAVs + 2 users

    # UAV positions
    positions[1 : 1 + NUM_UAVS, 0] = np.linspace(0.0, NUM_UAVS * SPACING, NUM_UAVS)
    positions[1 : 1 + NUM_UAVS, 1] = np.linspace(0.0, NUM_UAVS * SPACING, NUM_UAVS)
    positions[1 : 1 + NUM_UAVS, 2] = 50.0

    # User 0 position
    positions[1 + NUM_UAVS] = [10.0, 10.0, 0.0]

    # User 1 position
    positions[2 + NUM_UAVS] = [NUM_UAVS * SPACING, NUM_UAVS * SPACING, 0.0]

    net_sim.set_node_positions(positions)
    time.sleep(1.0)
    net_sim.verify_node_positions()
    print("Node positions updated and verified.")

    return net_sim


net_sim = setup_network()

user0_id = net_sim.get_node_id_from_type_id("user", 0)
user1_id = net_sim.get_node_id_from_type_id("user", 1)
user0_addr = net_sim.get_node_address(user0_id)
user1_addr = net_sim.get_node_address(user1_id)

t0 = time.time()
net_sim._update_ns3_init_time()
next_tx_time = 0.0
packet_count = 0
received_ids = set()
keep_sending = True

while time.time() - t0 < 30.0:  # Run for 30 seconds
    t_sim = time.time() - t0

    if keep_sending and t_sim > 25.0:
        # Stop sending packets after 25 seconds
        keep_sending = False
        print("Stopping packet transmission...")

    # Send a packet if it's time
    if keep_sending and t_sim >= next_tx_time:
        packet_count += 1
        msg = f"{packet_count}${t_sim}"
        packet = SimPacket(
            node_id=user0_id,
            src_addr=user0_addr,
            dst_addr=user1_addr,
            data=msg.encode(),
        )
        net_sim.send_packet(packet)
        print(f"[TX] Packet {packet_count} sent at {t_sim:.2f} s")
        next_tx_time = t_sim + np.random.normal(TX_INTERVAL, TX_INTERVAL / 10)

    # Fetch and process received packets
    net_sim.fetch_packets()
    packets = net_sim.get_node_packets(user1_id, delete=True)
    for packet in packets:
        msg = packet.data.decode()
        packet_id_str, tx_time_str = msg.split("$")
        packet_id = int(packet_id_str)
        tx_time = float(tx_time_str)
        rtt_global = t_sim - tx_time
        rtt_ns3 = packet.egress_time - packet.ingress_time
        print(
            f"[RX] Packet {packet_id} received, "
            f"Global RTT: {rtt_global:.3f} s, "
            f"NS-3 RTT: {rtt_ns3:.3f} s"
        )
        received_ids.add(packet_id)

# Report packet statistics
lost_packets = packet_count - len(received_ids)
print(f"Total packets sent:     {packet_count}")
print(f"Total packets received: {len(received_ids)}")
print(f"Total packets lost:     {lost_packets}")

print(f"Simulation time: {t_sim:.2f} s")
net_sim._update_ns3_last_time()
print(f"NS-3 elapsed time: {net_sim.elapsed_time:.2f} s")

net_sim.shutdown_simulator()
