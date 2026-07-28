import unittest

import numpy as np

from sim.environment import Environment
from sim.sdqn.environment import cartesian_frame_factory, default_frame_factory
from sim.sdqn.frames import FrameLayer, LogPolarGeometry, ScenarioState, SquareGeometry


class ProbeLayer(FrameLayer):
    def build_frame(self, state: ScenarioState) -> np.ndarray:
        return np.zeros(self.geometry.shape)


class FrameLayerTests(unittest.TestCase):
    def test_ground_cells_are_relative_to_the_observing_drone(self) -> None:
        environment = Environment()
        geometry = SquareGeometry(side_size=2, radius=2.0)
        layer = ProbeLayer(geometry, environment)
        state = ScenarioState(
            agent_position=np.array([10.0, 20.0, 30.0]),
            neighbor_positions=np.zeros((0, 3)),
            user_positions=np.zeros((0, 3)),
        )

        absolute = layer.absolute_cell_ground_positions(state)
        relative = layer.relative_cell_ground_positions(state)

        np.testing.assert_allclose(
            absolute[:, :2],
            geometry.flat_cell_positions + state.agent_position[:2],
        )
        np.testing.assert_array_equal(absolute[:, 2], 0.0)
        np.testing.assert_allclose(relative, absolute - state.agent_position)

    def test_logpolar_geometry_discards_radially_out_of_range_points(self) -> None:
        geometry = LogPolarGeometry(
            num_radial=4,
            num_angular=8,
            min_radius=10.0,
            max_radius=100.0,
        )
        positions = np.array([[1.0, 0.0], [20.0, 0.0], [1_000.0, 0.0]])

        indices = geometry.positions_to_cell_indices(positions, clip=False)

        self.assertEqual(indices.shape, (1, 2))
        np.testing.assert_array_equal(
            geometry.positions_to_cell_indices(positions, clip=True)[:, 0],
            np.array([0, 1, 3]),
        )

    def test_supported_observation_factories_share_model_shape(self) -> None:
        cartesian = cartesian_frame_factory()
        logpolar = default_frame_factory()

        self.assertEqual(cartesian.shape, (84, 84, 2))
        self.assertEqual(logpolar.shape, (84, 84, 2))
        self.assertIsInstance(cartesian.geometry_factory.create(), SquareGeometry)
        self.assertIsInstance(logpolar.geometry_factory.create(), LogPolarGeometry)


if __name__ == "__main__":
    unittest.main()
