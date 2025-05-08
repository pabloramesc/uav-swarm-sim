import subprocess
import time
from swarm_sim.network.sim_bridge import SimBridge, SimPacket

import numpy as np

# Run update_net_sim.sh at the beginning
subprocess.run(["sh", "update_code.sh"], cwd="./network_sim", check=True)

# Launch the NS3 simulation
ns3_process = subprocess.Popen(
    ["./ns3", "run", "scratch/swarm-net-sim/main --nGCS=2 --nUAV=3 --nUser=4"],
    cwd="./network_sim/ns-3",
)

bridge = SimBridge()
time.sleep(5.0)

while not bridge.is_ns3_running():
    time.sleep(5.0)

print("NS-3 process is running.")
time.sleep(1.0)

bridge.set_node_positions(
    {
        0: [0.0, 0.0, 0.0],
        1: [10.0, 10.0, 10.0],
        2: [20.0, 20.0, 20.0],
        3: [30.0, 30.0, 30.0],
        4: [40.0, 40.0, 40.0],
        5: [50.0, 50.0, 50.0],
    }
)
time.sleep(1.0)

positions = bridge.get_node_positions()
for node_id, node_pos in positions.items():
    print(f"Node {node_id} position: {node_pos}")
time.sleep(1.0)

addresses = bridge.get_node_addresses()
for node_id, ip_addr in addresses.items():
    print(f"Node {node_id} has IP address {ip_addr}")

bridge.send_ingress_packet(
    SimPacket(node_id=0, src_addr="0.0.0.0", dst_addr="10.0.2.1", data=b"Hello NS-3!")
)
bridge.send_ingress_packet(
    SimPacket(node_id=0, src_addr="0.0.0.0", dst_addr="10.0.2.2", data=b"Hello NS-3!")
)
bridge.send_ingress_packet(
    SimPacket(node_id=0, src_addr="0.0.0.0", dst_addr="10.0.2.3", data=b"Hello NS-3!")
)
time.sleep(1.0)

bridge.read_egress_packets()
time.sleep(1.0)

bridge.stop_ns3()
time.sleep(1.0)

# Ensure the NS-3 process is terminated
if ns3_process.poll() is None:  # Check if the process is still running
    print("NS-3 process is still running. Terminating...")
    ns3_process.terminate()  # Send termination signal
    ns3_process.wait()  # Wait for the process to terminate
    print("NS-3 process terminated successfully.")
else:
    print("NS-3 process already terminated.")
