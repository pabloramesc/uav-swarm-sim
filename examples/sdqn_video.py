"""Render a finite SDQN inference or observation-preview run to GIF or MP4."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from sim.sdqn import (
    SDQNEnvironmentConfig,
    SDQNSimulator,
    SeededRandomPolicy,
    cartesian_frame_factory,
    default_frame_factory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--model", type=Path)
    policy.add_argument(
        "--random-policy",
        action="store_true",
        help="use reproducible random actions for an observation preview",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/sdqn.mp4"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--drones", type=int, default=16)
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--height", type=float, default=10.0)
    parser.add_argument("--boundary-size", type=float, default=2_000.0)
    parser.add_argument("--obstacles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--representation",
        choices=("logpolar", "cartesian"),
        default="logpolar",
        help=(
            "observation geometry to render; it must match --model when supplied "
            "(default: logpolar)"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.model is not None and not args.model.is_file():
        raise FileNotFoundError(f"SDQN model does not exist: {args.model}")
    if args.duration <= 0.0 or args.fps <= 0.0:
        raise ValueError("--duration and --fps must be positive.")
    if args.output.suffix.lower() not in {".gif", ".mp4"}:
        raise ValueError("--output must use a .gif or .mp4 extension.")

    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    from sim.gui.sdqn_viewer import SDQNViewer

    config = SDQNEnvironmentConfig(
        dt=args.dt,
        num_drones=args.drones,
        num_users=args.users,
        drones_speed=args.speed,
        drones_height=args.height,
        boundary_size=args.boundary_size,
        num_obstacles=args.obstacles,
    )
    frame_factory = (
        cartesian_frame_factory()
        if args.representation == "cartesian"
        else default_frame_factory()
    )
    simulator = SDQNSimulator(
        config=config,
        model_path=args.model,
        policy=SeededRandomPolicy(seed=args.seed) if args.random_policy else None,
        frame_factory=frame_factory,
    )
    simulator.reset(seed=args.seed)
    viewer = SDQNViewer(simulator, fps=args.fps, background_type="rssi")
    viewer.reset()

    capture_period = 1.0 / args.fps
    next_capture = capture_period
    steps = math.ceil(args.duration / config.dt)
    frames_written = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".gif":
        writer = imageio.get_writer(
            args.output,
            mode="I",
            duration=1_000.0 / args.fps,
            loop=0,
        )
    else:
        writer = imageio.get_writer(args.output, fps=args.fps, format="ffmpeg")
    try:
        for _ in range(steps):
            simulator.step()
            if simulator.time + 1e-12 >= next_capture:
                viewer.render(force=True)
                frame = viewer.capture_frame()
                while simulator.time + 1e-12 >= next_capture:
                    writer.append_data(frame)
                    frames_written += 1
                    next_capture += capture_period
            print(simulator.simulation_status_str, end="\r")
        if frames_written == 0:
            viewer.render(force=True)
            writer.append_data(viewer.capture_frame())
        print()
    finally:
        writer.close()
        viewer.close()
        simulator.close()

    return args.output


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model is not None and not args.model.is_file():
        parser.error(f"--model must point to an existing file: {args.model}")
    output = run(args)
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
