from swarm_sim.network.network_simulator import NetworkSimulator, SimPacket

import time

NUM_UAVS = 10
NUM_USERS = 5

net_sim = NetworkSimulator(num_gcs=1, num_uavs=NUM_UAVS, num_users=NUM_USERS)
net_sim.launch_simulator()

node_id = 0
gcs_addr = net_sim.node_id_to_ip_address(node_id)

for uav_id in range(NUM_UAVS):
    node_id += 1
    uav_addr = net_sim.node_id_to_ip_address(node_id)
    msg_data = f"Hello from GCS (node {0}) to UAV {uav_id} (node ({node_id}))".encode("utf-8")
    packet = SimPacket(node_id=0, src_addr=gcs_addr, dst_addr=uav_addr, data=msg_data)
    net_sim.send_packet(packet)

for user_id in range(NUM_USERS):
    node_id += 1
    uav_addr = net_sim.node_id_to_ip_address(node_id)
    msg_data = f"Hello from GCS (node {0}) to user {user_id} (node ({node_id}))".encode("utf-8")
    packet = SimPacket(node_id=0, src_addr=gcs_addr, dst_addr=uav_addr, data=msg_data)
    net_sim.send_packet(packet)
    
time.sleep(5.0)
    
net_sim.fetch_packets()
for id in range(NUM_UAVS + NUM_USERS + 1):
    packets = net_sim.get_node_packets(id, delete=True)
    print(f"Node {id} Rx packets:", packets)

net_sim.stop_simulator()