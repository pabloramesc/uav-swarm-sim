from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from sim.environment import Environment  # noqa: E402
from sim.evsm import EVSMConfig, EVSMSimulator  # noqa: E402
from sim.gui.evsm_viewer import EVSMViewer  # noqa: E402


class EVSMSimulatorSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        environment = Environment()
        environment.set_rectangular_boundary((0.0, 0.0), (200.0, 200.0))

        config = EVSMConfig(
            separation_distance=50.0,
            obstacle_distance=5.0,
            target_altitude=20.0,
            initial_natural_length=5.0,
            natural_length_rate=1.0,
            control_update_period=0.01,
            springs_update_period=0.01,
        )
        self.simulator = EVSMSimulator(
            environment=environment,
            num_gcs=1,
            num_drones=4,
            num_users=2,
            config=config,
            dt=0.01,
            use_network=False,
            seed=3,
            metrics_area_samples=32,
        )
        self.addCleanup(self.simulator.close)

    def test_reset_step_metrics_and_headless_viewer(self) -> None:
        snapshot = self.simulator.reset(
            home=(40.0, 40.0),
            spacing=5.0,
            altitude=20.0,
        )

        self.assertEqual(self.simulator.num_agents, 7)
        self.assertFalse(hasattr(self.simulator, "sim"))
        self.assertEqual(
            [agent.agent_id for agent in self.simulator.all_agents],
            list(range(7)),
        )
        self.assertEqual(snapshot.step_count, 0)
        np.testing.assert_allclose(self.simulator.drone_states[:, 2], 20.0)
        self.assertEqual(self.simulator.evsm_monitor.edge_mask.shape, (4,))
        self.assertEqual(self.simulator.evsm_monitor.springs_matrix.shape, (4, 4))
        self.assertIsNotNone(self.simulator.metrics)

        for _ in range(5):
            snapshot = self.simulator.step()

        self.assertEqual(snapshot.step_count, 5)
        self.assertAlmostEqual(snapshot.time, 0.05)
        self.assertTrue(np.isfinite(self.simulator.drone_states).all())
        self.assertIsNotNone(self.simulator.metrics)

        viewer = EVSMViewer(
            self.simulator,
            background_type="none",
            show_legend=True,
        )
        self.addCleanup(plt.close, viewer.fig)
        viewer.render(force=True)
        frame = viewer.capture_frame()
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(
            len(viewer.spring_lines.get_xdata()),
            len(viewer.spring_lines.get_ydata()),
        )
        self.assertGreater(len(viewer.spring_lines.get_xdata()), 0)

    def test_metrics_sampling_does_not_change_user_motion(self) -> None:
        environment = Environment()
        environment.set_rectangular_boundary((0.0, 0.0), (200.0, 200.0))
        config = EVSMConfig(
            separation_distance=50.0,
            obstacle_distance=5.0,
            target_altitude=20.0,
            initial_natural_length=5.0,
        )
        simulators = [
            EVSMSimulator(
                environment=environment,
                num_gcs=1,
                num_drones=1,
                num_users=2,
                config=config,
                seed=11,
                metrics_area_samples=samples,
            )
            for samples in (0, 100)
        ]
        for simulator in simulators:
            self.addCleanup(simulator.close)
            simulator.reset(home=(40.0, 40.0), altitude=20.0)
            simulator.step()

        np.testing.assert_allclose(
            simulators[0].user_states,
            simulators[1].user_states,
        )


if __name__ == "__main__":
    unittest.main()
