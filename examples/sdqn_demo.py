"""Run a finite SDQN inference simulation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sim.sdqn import SDQNEnvironmentConfig, SDQNSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--drones", type=int, default=16)
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=10.0)
    parser.add_argument("--boundary-size", type=float, default=2_000.0)
    parser.add_argument("--obstacles", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the simulation and observation channels",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(f"SDQN model does not exist: {args.model}")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    config = SDQNEnvironmentConfig(
        dt=args.dt,
        num_drones=args.drones,
        num_users=args.users,
        drones_speed=args.speed,
        drones_height=args.height,
        boundary_size=args.boundary_size,
        num_obstacles=args.obstacles,
    )
    simulator = SDQNSimulator(config=config, model_path=args.model)
    simulator.reset(seed=args.seed)

    viewer = None
    if args.render:
        from sim.gui.sdqn_viewer import SDQNViewer

        viewer = SDQNViewer(simulator, fps=10.0, background_type="rssi")
        viewer.reset()

    try:
        for _ in range(args.steps):
            simulator.step()
            if viewer is not None:
                viewer.render()
            print(simulator.simulation_status_str, end="\r")
        print()
    finally:
        if viewer is not None:
            viewer.close()
        simulator.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f"--model must point to an existing file: {args.model}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
