from multiagent_sim.network.network_simulator import NetworkSimulator, SimPacket
import time
import numpy as np
import csv
import os

# Configuration
UAV_SPACING = 100.0
UAV_ALTITUDE = 50.0

TX_INTERVAL = 0.1
PRE_TX_DELAY = 0.0
TX_DURATION = 100.0
POST_TX_DELAY = 0.0
SIM_DURATION = PRE_TX_DELAY + TX_DURATION + POST_TX_DELAY

OUTPUT_FILE = "rtt_results_v4.csv"


def setup_network(grid_size: int, spacing: float) -> NetworkSimulator:
    """Initialize simulator and place users in two corners and drones in a grid NxN."""
    num_uavs = grid_size ** 2
    net = NetworkSimulator(num_gcs=0, num_drones=num_uavs, num_users=2)
    net.launch_simulator(max_attempts=10)

    total_nodes = 0 + num_uavs + 2  # GCS + drones + users
    positions = np.zeros((total_nodes, 3))

    # Place drones in a grid NxN on the XY plane, altitude fixed
    for i in range(grid_size):
        for j in range(grid_size):
            idx = 1 + i * grid_size + j  # drone node id
            positions[idx] = [i * spacing, j * spacing, UAV_ALTITUDE]

    # Users at opposite corners of the grid (ground level)
    positions[num_uavs + 0] = [0.0, 0.0, 0.0]                          # user0 at bottom-left corner
    positions[num_uavs + 1] = [(grid_size - 1) * spacing,             # user1 at top-right corner
                              (grid_size - 1) * spacing,
                              0.0]

    net.set_node_positions(positions={id: pos for id, pos in enumerate(positions)})
    net.verify_node_positions()
    print(f"✔ Node positions set for {num_uavs} UAVs in a {grid_size}x{grid_size} grid")
    return net


def run_experiment(grid_size: int):
    net = setup_network(grid_size, UAV_SPACING)

    user0 = net.get_node_from_name("user0")
    user1 = net.get_node_from_name("user1")

    sim_t0 = time.time()
    ns3_t0 = net.ns3_time

    packet_id = 0
    next_tx_time = 0.0
    received_ids = set()
    rtts_global, rtts_ns3 = [], []

    print(f"[INFO] Starting simulation loop for {SIM_DURATION} seconds...")

    while (now := time.time() - sim_t0) < SIM_DURATION:
        # Transmit
        if PRE_TX_DELAY <= now <= PRE_TX_DELAY + TX_DURATION and now >= next_tx_time:
            packet_id += 1
            msg = f"{packet_id}${now}"
            packet = SimPacket(
                node_id=user0.node_id,
                src_addr=user0.addr,
                dst_addr=user1.addr,
                data=msg.encode(),
            )
            net.send_packet(packet)
            next_tx_time = now + np.random.normal(TX_INTERVAL, 0)

        # Receive
        net.fetch_packets()
        for pkt in net.get_node_packets(user1.node_id, delete=True):
            msg = pkt.data.decode()
            pid_str, tx_time_str = msg.split("$")
            pid = int(pid_str)
            tx_time = float(tx_time_str)

            rtt_glob = now - tx_time
            rtt_ns3 = pkt.egress_time - pkt.ingress_time
            received_ids.add(pid)
            rtts_global.append(rtt_glob)
            rtts_ns3.append(rtt_ns3)
            
        # Print progress
        print(
            f"\r[PROGRESS] Time: {now:.2f}s | Sent: {packet_id} | Received: {len(received_ids)}",
            end="",
            flush=True,
        )

    print(f"\n[INFO] Shutting down simulator...")
    
    net.update_sim_time()
    sim_time = time.time() - sim_t0
    ns3_time = net.ns3_time - ns3_t0

    net.shutdown_simulator()
    del(net)

    received = len(received_ids)
    lost = packet_id - received

    # RTT statistics
    def stats(arr):
        arr = np.array(arr)
        return {
            "avg": np.mean(arr) if arr.size else np.nan,
            "min": np.min(arr) if arr.size else np.nan,
            "max": np.max(arr) if arr.size else np.nan,
            "std": np.std(arr) if arr.size else np.nan,
        }

    rtt_global_stats = stats(rtts_global)
    rtt_ns3_stats = stats(rtts_ns3)

    print(f"[SUMMARY] UAVs={grid_size**2} | Sent={packet_id} | Received={received} | Lost={lost}")
    print(
        f"          RTT Global Avg: {rtt_global_stats['avg']:.4f} s | "
        f"Min: {rtt_global_stats['min']:.4f} s | "
        f"Max: {rtt_global_stats['max']:.4f} s | "
        f"Std: {rtt_global_stats['std']:.4f} s"
    )
    print(
        f"          RTT NS3   Avg: {rtt_ns3_stats['avg']:.4f} s | "
        f"Min: {rtt_ns3_stats['min']:.4f} s | "
        f"Max: {rtt_ns3_stats['max']:.4f} s | "
        f"Std: {rtt_ns3_stats['std']:.4f} s"
    )
    print(f"          Real elapsed time: {sim_time:.2f} s | NS-3 elapsed time: {ns3_time:.2f} s")
    if sim_time - ns3_time > 1.0:
        print(f"⚠️  WARNING: Wall time exceeds simulated time by {sim_time - ns3_time:.2f} s")

    return {
        "num_uavs": grid_size ** 2,
        "sent": packet_id,
        "received": received,
        "lost": lost,
        "rtt_global_avg": rtt_global_stats["avg"],
        "rtt_global_min": rtt_global_stats["min"],
        "rtt_global_max": rtt_global_stats["max"],
        "rtt_global_std": rtt_global_stats["std"],
        "rtt_ns3_avg": rtt_ns3_stats["avg"],
        "rtt_ns3_min": rtt_ns3_stats["min"],
        "rtt_ns3_max": rtt_ns3_stats["max"],
        "rtt_ns3_std": rtt_ns3_stats["std"],
        "sim_time": sim_time,
        "ns3_time": ns3_time,
    }


def save_results(results, filename):
    write_header = not os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_uavs",
                "sent",
                "received",
                "lost",
                "rtt_global_avg",
                "rtt_global_min",
                "rtt_global_max",
                "rtt_global_std",
                "rtt_ns3_avg",
                "rtt_ns3_min",
                "rtt_ns3_max",
                "rtt_ns3_std",
                "sim_time",
                "ns3_time",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(results)


if __name__ == "__main__":
    for N in range(2, 10+1):  # Grid sizes from 2x2 to 10x10
        for _ in range(10):
            print(f"\n=== Running simulation with grid size {N}x{N} ({N**2} UAVs) ===")
            result = run_experiment(N)
            save_results(result, OUTPUT_FILE)
            print(f"✔ Results for grid size {N}x{N} saved.")
