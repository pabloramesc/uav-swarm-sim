"""Train the maintained vectorized SDQN policy."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from sim.sdqn import (
    DQNConfig,
    SDQNEnvironmentConfig,
    SDQNTrainer,
    cartesian_frame_factory,
    default_frame_factory,
)
from sim.sdqn.rewards import RewardConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("data/models/sdqn_model.keras")
    )
    parser.add_argument("--log", type=Path, default=Path("data/logs/sdqn_model.csv"))
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--episode-steps", type=int, default=1_000)
    parser.add_argument("--drones", type=int, default=2)
    parser.add_argument("--users", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--representation",
        choices=("logpolar", "cartesian"),
        default="logpolar",
        help="observation geometry to train (default: logpolar)",
    )
    parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def run(args: argparse.Namespace) -> None:
    environment_config = SDQNEnvironmentConfig(
        dt=1.0,
        num_drones=args.drones,
        num_users=args.users,
        drones_speed=2.0,
        drones_height=10.0,
        boundary_size=2_000.0,
        num_obstacles=0,
        reward=RewardConfig(
            collision_dist=1.0,
            users_coverage="difference",
            weight_users_coverage=1.0,
            collision_penalty=-1.0,
        ),
    )
    dqn_config = DQNConfig(
        memory_size=200_000,
        min_memory=50_000,
        update_freq=10_000,
        batch_size=32,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=1e-5,
        decay_type="linear",
        n_step=3,
        per_alpha=0.6,
        per_beta=0.4,
        per_beta_annealing=0.0,
        autosave_freq=10_000,
        learning_rate=1e-4,
    )
    trainer = SDQNTrainer(
        environment_config=environment_config,
        dqn_config=dqn_config,
        model_path=args.model,
        log_path=args.log,
        render=args.render,
        frame_factory=(
            cartesian_frame_factory()
            if args.representation == "cartesian"
            else default_frame_factory()
        ),
    )
    try:
        trainer.train(
            train_freq=1,
            max_episodes=args.episodes,
            max_episode_steps=args.episode_steps,
            seed=args.seed,
            verbose=True,
        )
    finally:
        trainer.close()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
