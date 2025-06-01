from multiagent_sim.network.swarm_link import SwarmLink
from multiagent_sim.network.network_simulator import NetworkSimulator, SimPacket
import time
import numpy as np

net_sim = NetworkSimulator(num_gcs=0, num_drones=2, num_users=0, verbose=True)

link0 = SwarmLink(
    agent_id=0,
    network_sim=net_sim,
    local_bcast_interval=0.1,
    global_bcast_interval=1.0,
    position_timeout=5.0,
)

link1 = SwarmLink(
    agent_id=1,
    network_sim=net_sim,
    local_bcast_interval=0.1,
    global_bcast_interval=1.0,
    position_timeout=5.0,
)


net_sim.launch_simulator(max_attempts=2)

t0, t = time.time(), 0.0
prev_pos_time = 0.0
while t < 60.0:
    t = time.time() - t0
    
    position0 = np.zeros(3)
    position1 = np.random.uniform(-100, 100, size=3)
    
    net_sim.update(positions={0: position0, 1: position1}, check=False)

    link0.update(time=t, position=position0)
    link1.update(time=t, position=position1)
    
    for drone_id, pos in link0.position_provider.neighbors.items():
        if pos.time > prev_pos_time:
            prev_pos_time = pos.time
            print(f"At {t:.2f}s, Drone {drone_id} position: {pos.position}, time: {pos.time}")
            if np.allclose(pos.position, position0, atol=1e-3):
                print(f"Warning: Drone {drone_id} position mismatch! Expected {position0}, got {pos.position}")
    

net_sim.shutdown_simulator()