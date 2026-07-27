import unittest
from dataclasses import dataclass

import numpy as np

from sim.environment import Environment
from sim.sdqn.actions import Action
from sim.sdqn.environment import SDQNEnvironment, SDQNEnvironmentConfig
from sim.sdqn.frames import (
    FrameGeneratorFactory,
    FrameLayer,
    FrameLayerFactory,
    SquareGeometryFactory,
)
from sim.sdqn.frames.geometry import FrameGeometry
from sim.sdqn.frames.state import ScenarioState
from sim.sdqn.rewards import RewardConfig


class AbsoluteXLayer(FrameLayer):
    """Encode absolute X in every cell, making stale frames easy to detect."""

    def __init__(self, geometry: FrameGeometry):
        super().__init__(geometry, plot_center=False)

    def build_frame(self, state: ScenarioState) -> np.ndarray:
        value = state.agent_position[0] / 100.0
        return np.full(self.geometry.shape, value, dtype=np.float32)


@dataclass
class AbsoluteXLayerFactory(FrameLayerFactory):
    def create(self, geo: FrameGeometry, env: Environment | None = None) -> FrameLayer:
        return AbsoluteXLayer(geo)


def small_frame_factory() -> FrameGeneratorFactory:
    return FrameGeneratorFactory(
        geometry_factory=SquareGeometryFactory(side_size=5, radius=10.0),
        layer_factories=[AbsoluteXLayerFactory()],
    )


def fixed_environment() -> Environment:
    environment = Environment(obstacles=[])
    environment.set_rectangular_boundary((0.0, 0.0), (100.0, 100.0))
    return environment


class SDQNEnvironmentTests(unittest.TestCase):
    def make_environment(self, *, collision_penalty: float = -1.0) -> SDQNEnvironment:
        config = SDQNEnvironmentConfig(
            dt=0.2,
            num_drones=1,
            num_users=0,
            drones_speed=5.0,
            drones_height=10.0,
            reward=RewardConfig(
                collision_dist=0.0,
                users_coverage=None,
                collision_penalty=collision_penalty,
            ),
        )
        return SDQNEnvironment(
            config,
            environment=fixed_environment(),
            frame_factory=small_frame_factory(),
        )

    @staticmethod
    def state_at(x: float, y: float = 50.0) -> np.ndarray:
        states = np.zeros((1, 6), dtype=np.float64)
        states[0, :3] = (x, y, 10.0)
        return states

    def test_reset_and_step_return_matching_frame_snapshot_and_caches(self) -> None:
        environment = self.make_environment()
        frames, info = environment.reset(
            seed=7,
            options={"drone_states": self.state_at(50.0)},
        )

        self.assertEqual(frames.shape, (1, 5, 5, 1))
        self.assertEqual(frames.dtype, np.uint8)
        np.testing.assert_array_equal(frames, environment.last_frames)
        np.testing.assert_allclose(environment.drone_states[0, :3], (50, 50, 10))
        self.assertEqual(environment.step_count, 0)
        self.assertEqual(info["snapshot"].step_count, 0)

        frames, rewards, terminated, truncated, info = environment.step(
            np.array([Action.RIGHT], dtype=np.int32)
        )

        # speed * dt = 5 * 0.2 = exactly 1 metre, applied once.
        expected_state = np.array([51.0, 50.0, 10.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(environment.drone_states[0], expected_state)
        np.testing.assert_allclose(environment.drones[0].state, expected_state)
        np.testing.assert_allclose(info["snapshot"].drone_states[0], expected_state)
        self.assertEqual(environment.step_count, 1)
        self.assertAlmostEqual(environment.time, 0.2)

        expected_pixel = np.uint8(0.51 * 255.0)
        np.testing.assert_array_equal(
            frames[0, ..., 0], np.full((5, 5), expected_pixel)
        )
        np.testing.assert_array_equal(frames, environment.last_frames)
        np.testing.assert_array_equal(rewards, np.array([0.0], dtype=np.float32))
        self.assertFalse(terminated.any())
        self.assertFalse(truncated.any())
        self.assertIs(info["metrics"], environment.metrics)

    def test_collision_attempt_is_reverted_before_single_core_step(self) -> None:
        environment = self.make_environment(collision_penalty=-7.0)
        initial = self.state_at(99.5)
        initial_frames, _ = environment.reset(options={"drone_states": initial})

        frames, rewards, terminated, truncated, info = environment.step(
            np.array([Action.RIGHT], dtype=np.int32)
        )

        np.testing.assert_allclose(environment.drone_states[0], initial[0])
        np.testing.assert_allclose(info["snapshot"].drone_states[0], initial[0])
        np.testing.assert_array_equal(frames, initial_frames)
        np.testing.assert_array_equal(rewards, np.array([-7.0], dtype=np.float32))
        np.testing.assert_array_equal(terminated, np.array([True]))
        self.assertFalse(truncated.any())
        self.assertEqual(environment.step_count, 1)
        self.assertAlmostEqual(environment.time, 0.2)

    def test_explicit_environment_is_not_randomized_on_reset(self) -> None:
        simulation = self.make_environment()
        boundary = simulation.environment.boundary
        simulation.reset(
            seed=4,
            options={"drone_states": self.state_at(50.0)},
        )
        self.assertIs(simulation.environment.boundary, boundary)
        self.assertEqual(simulation.environment.boundary.bounds.xlim, (0.0, 100.0))
        with self.assertRaisesRegex(ValueError, "Injected environments"):
            simulation.reset(options={"randomize_environment": False})

    def test_configuration_rejects_invalid_counts_and_ranges(self) -> None:
        with self.assertRaisesRegex(TypeError, "num_drones must be an integer"):
            SDQNEnvironmentConfig(num_drones=1.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "minimum cannot exceed"):
            SDQNEnvironmentConfig(boundary_size=(100.0, 10.0))
        with self.assertRaisesRegex(ValueError, "at least 0"):
            SDQNEnvironmentConfig(num_obstacles=(0, -1))

    def test_metrics_sampling_does_not_change_seeded_motion(self) -> None:
        simulations = [
            SDQNEnvironment(
                SDQNEnvironmentConfig(
                    dt=0.2,
                    num_drones=1,
                    num_users=1,
                    metrics_area_samples=samples,
                ),
                environment=fixed_environment(),
                frame_factory=small_frame_factory(),
            )
            for samples in (0, 100)
        ]
        for simulation in simulations:
            self.addCleanup(simulation.close)
            simulation.reset(seed=13)
            simulation.step(np.array([Action.NOP]))

        np.testing.assert_allclose(
            simulations[0].user_states,
            simulations[1].user_states,
        )


if __name__ == "__main__":
    unittest.main()
