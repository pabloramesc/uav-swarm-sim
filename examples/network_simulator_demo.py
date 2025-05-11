from multiagent_sim.network.network_simulator import NetworkSimulator, SimPacket

import time
import numpy as np

NUM_UAVS = 10
NUM_USERS = 10
NODES_SPACING = 50.0
MAX_PACKETS = 5
WAIT_TIME = 0.01
INTERVAL = 1.0

net_sim = NetworkSimulator(num_gcs=1, num_uavs=NUM_UAVS, num_users=NUM_USERS)
net_sim._rewrite_ns3_code()
net_sim.launch_simulator(max_attempts=2)

print("Updating node positions...")
positions = np.zeros((1 + NUM_UAVS + NUM_USERS, 3))
positions[1 : 1 + NUM_UAVS, 0:3] = NODES_SPACING * np.linspace(
    np.ones(3), NUM_UAVS * np.ones(3), NUM_UAVS
)
positions[1 + NUM_UAVS :, 0:2] = NODES_SPACING * np.linspace(
    np.ones(2), NUM_USERS * np.ones(2), NUM_USERS
)
net_sim.set_node_positions(positions)
time.sleep(1.0)
net_sim.verify_node_positions()
print("Node positions updated and verified")

time.sleep(5.0)

for _ in range(MAX_PACKETS):
    node_id = 0
    gcs_addr = net_sim.get_node_address(node_id)

    for uav_id in range(NUM_UAVS):
        node_id += 1
        uav_addr = net_sim.get_node_address(node_id)
        msg = f"Hello from GCS (node {0}) to UAV {uav_id} (node ({node_id}))"
        packet = SimPacket(
            node_id=0, src_addr=gcs_addr, dst_addr=uav_addr, data=msg.encode("utf-8")
        )
        net_sim.send_packet(packet)
        print("Packet sent:", packet)
        time.sleep(WAIT_TIME)

    for user_id in range(NUM_USERS):
        node_id += 1
        uav_addr = net_sim.get_node_address(node_id)
        msg = f"Hello from GCS (node {0}) to user {user_id} (node ({node_id}))"
        packet = SimPacket(
            node_id=0, src_addr=gcs_addr, dst_addr=uav_addr, data=msg.encode("utf-8")
        )
        net_sim.send_packet(packet)
        print("Packet sent:", packet)
        time.sleep(WAIT_TIME)

    msg = f"Hello from GCS (node {0}) to all in local (10.0.255.255)"
    packet = SimPacket(
        node_id=0, src_addr=gcs_addr, dst_addr="10.0.255.255", data=msg.encode("utf-8")
    )
    net_sim.send_packet(packet)
    print("Packet sent:", packet)
    time.sleep(WAIT_TIME)

    msg = f"Hello from GCS (node {0}) to all in global (255.255.255.255)"
    packet = SimPacket(
        node_id=0,
        src_addr=gcs_addr,
        dst_addr="255.255.255.255",
        data=msg.encode("utf-8"),
    )
    net_sim.send_packet(packet)
    print("Packet sent:", packet)
    time.sleep(WAIT_TIME)
    
    time.sleep(INTERVAL)

net_sim.fetch_packets()
for id in range(NUM_UAVS + NUM_USERS + 1):
    print(f"Node {id} received packets:")
    packets = net_sim.get_node_packets(id, delete=True)
    for packet in packets:
        print("-", packet)

net_sim.shutdown_simulator()
