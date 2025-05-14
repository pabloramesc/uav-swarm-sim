import time

from multiagent_sim.network.network_interface import NetworkInterface
from multiagent_sim.network.network_simulator import NetworkSimulator, SimPacket

net_sim = NetworkSimulator(num_gcs=0, num_drones=2, num_users=0, verbose=True)

iface0 = NetworkInterface(
    node_id=0,
    network_sim=net_sim,
)

iface1 = NetworkInterface(
    node_id=1,
    network_sim=net_sim,
)

net_sim.launch_simulator(max_attempts=2)

packet = SimPacket(
    node_id=0,
    src_addr=iface0.node_address,
    dst_addr=iface1.node_address,
    data="Hello from Node 0!".encode("utf-8"),
)

iface0.send(packet)
print(f"Sent packet: {packet}")

time.sleep(1.0)

net_sim.fetch_packets()
received_packets = iface1.receive()
print(f"Received packets: {received_packets}")

time.sleep(1.0)

net_sim.shutdown_simulator()