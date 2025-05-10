import subprocess
import time
from multiagent_sim.network.sim_bridge import SimBridge, SimPacket

import numpy as np

# Run update_net_sim.sh at the beginning
subprocess.run(["sh", "update_code.sh"], cwd="./network_sim", check=True)

# Launch the NS3 simulation
print("Starting NS-3 simulation...")
ns3_process = subprocess.Popen(
    ["./ns3", "run", "scratch/swarm-net-sim/main --nGCS=2 --nUAV=3 --nUser=4"],
    cwd="./network_sim/ns-3",
)
time.sleep(5.0)

bridge = SimBridge()
time.sleep(1.0)

# Wait until NS-3 responds to heartbeat
print("Waiting for NS-3 to start...")
while not bridge.is_ns3_running():
    print("NS-3 is not responding. Waiting for 1 second...")
    time.sleep(1.0)
print("NS-3 process is running.")

# Query and print the current NS-3 simulation time
sim_time = bridge.get_ns3_time()
if sim_time is not None:
    print(f"Initial NS-3 simulation time: {sim_time:.6f} s")
else:
    print("Failed to retrieve initial NS-3 simulation time.")
time.sleep(1.0)

# Set node positions
print("Setting node positions...")
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

# Get and print node positions
print("Getting node positions...")
positions = bridge.get_node_positions()
for node_id, node_pos in positions.items():
    print(f"- Node {node_id} position: {node_pos}")
time.sleep(1.0)

# Get and print node addresses
print("Getting node addresses...")
addresses = bridge.get_node_addresses()
for node_id, ip_addr in addresses.items():
    print(f"- Node {node_id} has IP address {ip_addr}")
    
# Send ingress packets
for dst in ["10.0.2.1", "10.0.2.2", "10.0.2.3"]:
    print(f"Sending packet to {dst}...")
    bridge.send_ingress_packet(
        SimPacket(node_id=0, src_addr="0.0.0.0", dst_addr=dst, data=b"Hello NS-3!")
    )
    time.sleep(1.0)

# Read and display egress packets
print("Reading egress packets...")
egress_packets = bridge.read_egress_packets()
print(f"Received {len(egress_packets)} egress packets:")
for packet in egress_packets:
    print(">>>", packet)

# Final simulation time
sim_time = bridge.get_ns3_time()
if sim_time is not None:
    print(f"Final NS-3 simulation time: {sim_time:.6f} s")
else:
    print("Failed to retrieve final NS-3 sim time.")

# Stop NS-3 and clean up
print("Stopping NS-3 simulation...")
bridge.stop_ns3()
time.sleep(1.0)

# Ensure the NS-3 process is terminated
if ns3_process.poll() is None:  # Still running?
    print("NS-3 process is still running. Terminating...")
    ns3_process.terminate()
    ns3_process.wait()
    print("NS-3 process terminated successfully.")
else:
    print("NS-3 process already terminated.")
