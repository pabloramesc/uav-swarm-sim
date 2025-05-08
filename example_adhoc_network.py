from swarm_sim.network.network_simulator import NetworkSimulator, SimPacket

import time
import numpy as np

NUM_UAVS = 10

net_sim = NetworkSimulator(num_gcs=1, num_uavs=NUM_UAVS, num_users=2)
# net_sim._rewrite_ns3_code()
net_sim.launch_simulator(max_attempts=2)

positions = np.zeros((1 + 10 + 2))
# UAV positions
positions[1 : 1 + NUM_UAVS, 0] = np.linspace(0.0, NUM_UAVS * 50.0, NUM_UAVS)
positions[1 : 1 + NUM_UAVS, 1] = np.linspace(0.0, NUM_UAVS * 50.0, NUM_UAVS)
positions[1 : 1 + NUM_UAVS, 2] = 50.0
# User 0 position
positions[1 + NUM_UAVS : 2 + NUM_UAVS, 0] = 10.0
positions[1 + NUM_UAVS : 2 + NUM_UAVS, 1] = 10.0
positions[1 + NUM_UAVS : 2 + NUM_UAVS, 2] = 0.0
# User 1 position
positions[2 + NUM_UAVS : 3 + NUM_UAVS, 0] = 10.0
positions[2 + NUM_UAVS : 3 + NUM_UAVS, 1] = 10.0
positions[2 + NUM_UAVS : 3 + NUM_UAVS, 2] = 0.0
