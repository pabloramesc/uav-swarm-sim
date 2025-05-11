from multiagent_sim.network.swarm_link import SwarmLink
from multiagent_sim.network.network_simulator import NetworkSimulator, SimPacket
import time
import numpy as np

net_sim = NetworkSimulator(num_gcs=0, num_uavs=2, num_users=0, verbose=True)

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
    
    net_sim.update()

    link0.update(time=t, position=np.zeros(3))
    link1.update(time=t, position=np.random.rand(3))
    
    for drone_id, pos in link0.drone_positions.items():
        if pos.time > prev_pos_time:
            prev_pos_time = pos.time
            print(f"At {t:.2f}s, Drone {drone_id} position: {pos.position}, time: {pos.time}")
    

net_sim.shutdown_simulator()