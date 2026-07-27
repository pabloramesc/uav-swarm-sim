"""Run a finite EVSM deployment over the Barcelona terrain dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from math import ceil
from pathlib import Path

from sim.environment import Environment
from sim.evsm import EVSMConfig, EVSMSimulator
from sim.gui.evsm_viewer import EVSMViewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = PROJECT_ROOT / "data" / "elevation" / "barcelona_dem.tif"


def main(duration: float = 120.0, *, fetch_satellite: bool = False) -> None:
    """Run the terrain scenario for ``duration`` simulated seconds."""

    if duration <= 0.0:
        raise ValueError("duration must be positive.")

    dt = 0.01
    grid_spacing = 100.0
    config = EVSMConfig(
        separation_distance=1_000.0,
        obstacle_distance=100.0,
        target_altitude=50.0,
        initial_natural_length=grid_spacing,
        natural_length_rate=10.0,
        max_acceleration=10.0,
        max_position_error=100.0,
    )

    environment = Environment(
        dem_path=str(DEM_PATH),
        fetch_satellite=fetch_satellite,
    )
    environment.set_polygonal_boundary(
        (
            (2_000.0, 2_000.0),
            (6_000.0, 2_000.0),
            (8_000.0, 4_000.0),
            (8_000.0, 8_000.0),
            (2_000.0, 8_000.0),
        )
    )
    environment.add_circular_obstacle((6_000.0, 4_000.0), 500.0)
    environment.add_rectangular_obstacle(
        (4_000.0, 5_500.0),
        (6_000.0, 6_500.0),
    )

    simulator = EVSMSimulator(
        environment=environment,
        num_drones=25,
        num_users=50,
        num_gcs=1,
        config=config,
        dt=dt,
        use_network=False,
        seed=7,
    )
    simulator.reset(
        home=(4_000.0, 4_000.0),
        spacing=grid_spacing,
        altitude=100.0,
    )
    viewer = EVSMViewer(simulator, background_type="fused")

    try:
        for _ in range(ceil(duration / dt)):
            simulator.step()
            viewer.render()
            print(
                f"Real time: {simulator.real_time:.2f} s, "
                f"sim time: {simulator.time:.2f} s, "
                f"FPS: {viewer.fps:.2f}",
                end="\r",
            )
    finally:
        viewer.close()
        simulator.close()
    print()


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument(
        "--satellite",
        action="store_true",
        help="download satellite tiles for the terrain background",
    )
    args = parser.parse_args(argv)
    main(duration=args.duration, fetch_satellite=args.satellite)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
