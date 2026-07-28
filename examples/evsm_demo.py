"""Run and log a finite EVSM deployment in a flat environment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from math import ceil
from pathlib import Path

import numpy as np

from sim.environment import Environment
from sim.evsm import EVSMConfig, EVSMSimulator
from sim.gui.evsm_viewer import EVSMViewer
from sim.utils.data_logger import DataLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(duration: float = 120.0, *, use_network: bool = False) -> None:
    """Run the EVSM example for ``duration`` simulated seconds."""

    if duration <= 0.0:
        raise ValueError("duration must be positive.")

    dt = 0.01
    num_drones = 25
    num_users = 10
    size = 1_000.0
    grid_spacing = 10.0

    config = EVSMConfig(
        separation_distance=350.0,
        obstacle_distance=20.0,
        max_acceleration=10.0,
        target_altitude=0.0,
        initial_natural_length=grid_spacing,
        natural_length_rate=5.0,
    )

    environment = Environment()
    environment.set_rectangular_boundary((0.0, 0.0), (size, size))
    environment.add_circular_obstacle(center=(600.0, 600.0), radius=100.0)
    environment.add_rectangular_obstacle((200.0, 600.0), (300.0, 800.0))
    environment.add_rectangular_obstacle((600.0, 200.0), (800.0, 300.0))

    simulator = EVSMSimulator(
        environment=environment,
        num_drones=num_drones,
        num_users=num_users,
        num_gcs=1,
        config=config,
        dt=dt,
        use_network=use_network,
        seed=7,
    )
    simulator.reset(
        home=(200.0, 200.0),
        spacing=grid_spacing,
        altitude=config.target_altitude,
    )

    viewer = EVSMViewer(simulator, show_legend=True)
    log = DataLogger(
        log_file="evsm_demo.npz",
        log_folder=str(PROJECT_ROOT / "logs"),
        columns=[
            "time",
            "area_cov",
            "users_cov",
            "direct_conn",
            "global_conn",
            "send_packets",
            "recv_packets",
        ],
    )

    try:
        for _ in range(ceil(duration / dt)):
            simulator.step(sync=use_network)
            viewer.render()

            send_packets, receive_packets = _packet_averages(simulator.users)
            metrics = simulator.metrics
            if metrics is None:
                raise RuntimeError("Metrics are unavailable before reset.")
            log.append(
                [
                    simulator.time,
                    metrics.area_coverage,
                    metrics.users_coverage,
                    metrics.direct_connections,
                    metrics.global_connections,
                    send_packets,
                    receive_packets,
                ]
            )

            print(
                f"Real time: {simulator.real_time:.2f} s, "
                f"sim time: {simulator.time:.2f} s, "
                f"FPS: {viewer.fps:.2f}",
                end="\r",
            )
    finally:
        log.dump()
        viewer.close()
        simulator.close()

    print(f"\nLog written to {log.log_file}")


def _packet_averages(users) -> tuple[float, float]:
    links = [
        user.swarm_link
        for user in users
        if getattr(user, "swarm_link", None) is not None
    ]
    if not links:
        return 0.0, 0.0
    return (
        float(np.mean([link.send_counter for link in links])),
        float(np.mean([link.recv_counter for link in links])),
    )


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument(
        "--network",
        action="store_true",
        help="use the optional NS-3 network backend",
    )
    args = parser.parse_args(argv)
    main(duration=args.duration, use_network=args.network)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
