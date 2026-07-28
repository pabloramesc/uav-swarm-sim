"""Render a finite EVSM simulation directly to an MP4 file."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from math import ceil, isfinite
from pathlib import Path

from sim.environment import Environment
from sim.evsm import EVSMConfig, EVSMSimulator
from sim.gui.evsm_viewer import EVSMViewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(
    duration: float = 120.0,
    *,
    video_fps: int = 10,
    speedup: float = 1.0,
    use_network: bool = False,
    output: str | Path | None = None,
) -> Path:
    """Generate an MP4 and return its absolute path."""

    if not isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be positive.")
    if video_fps <= 0:
        raise ValueError("video_fps must be positive.")
    if not isfinite(speedup) or speedup <= 0.0:
        raise ValueError("speedup must be positive.")

    import imageio.v2 as imageio

    dt = 0.01
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
    environment.set_rectangular_boundary((0.0, 0.0), (1_000.0, 1_000.0))
    environment.add_circular_obstacle((600.0, 600.0), 100.0)
    environment.add_rectangular_obstacle((200.0, 600.0), (300.0, 800.0))
    environment.add_rectangular_obstacle((600.0, 200.0), (800.0, 300.0))

    simulator = EVSMSimulator(
        environment=environment,
        num_drones=25,
        num_users=10,
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
    viewer = EVSMViewer(
        simulator,
        show_legend=True,
        fig_size=(9.6, 8.0),
    )

    mode = "network" if use_network else "ideal"
    video_path = (
        (
            PROJECT_ROOT / "outputs" / f"evsm_{mode}.mp4"
            if output is None
            else Path(output)
        )
        .expanduser()
        .resolve()
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)

    capture_period = speedup / video_fps
    next_capture_time = capture_period
    frames_written = 0
    writer = imageio.get_writer(video_path, fps=video_fps)
    try:
        for _ in range(ceil(duration / dt)):
            simulator.step(sync=use_network)
            if simulator.time + 1e-12 < next_capture_time:
                continue

            viewer.render(force=True)
            frame = viewer.capture_frame()
            while simulator.time + 1e-12 >= next_capture_time:
                writer.append_data(frame)
                frames_written += 1
                next_capture_time += capture_period
            print(
                f"Sim time: {simulator.time:.2f} s, frames: {frames_written}",
                end="\r",
            )
        if frames_written == 0:
            viewer.render(force=True)
            writer.append_data(viewer.capture_frame())
            frames_written = 1
    finally:
        writer.close()
        viewer.close()
        simulator.close()

    print(f"\nVideo written to {video_path}")
    return video_path


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="simulated seconds represented by each second of video",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--network",
        action="store_true",
        help="use the optional NS-3 network backend",
    )
    args = parser.parse_args(argv)
    main(
        duration=args.duration,
        video_fps=args.fps,
        speedup=args.speedup,
        use_network=args.network,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
