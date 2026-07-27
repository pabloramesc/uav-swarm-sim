from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from sim.environment import Environment  # noqa: E402
from sim.gui.sdqn_viewer import SDQNViewer  # noqa: E402
from sim.gui.simple_viewer import SimpleViewer  # noqa: E402
from sim.sdqn.environment import (  # noqa: E402
    SDQNEnvironment,
    SDQNEnvironmentConfig,
)


def make_environment() -> SDQNEnvironment:
    world = Environment()
    world.set_rectangular_boundary((0.0, 0.0), (100.0, 100.0))
    simulation = SDQNEnvironment(
        SDQNEnvironmentConfig(num_drones=1, num_users=0),
        environment=world,
    )
    drone_states = np.array([[50.0, 50.0, 10.0, 0.0, 0.0, 0.0]])
    simulation.reset(options={"drone_states": drone_states})
    return simulation


class ViewerInitializationTests(unittest.TestCase):
    def test_simple_viewer_builds_legend_after_plot_artists(self) -> None:
        simulation = make_environment()
        self.addCleanup(simulation.close)
        viewer = SimpleViewer(
            simulation,
            background_type="none",
            show_legend=True,
        )
        self.addCleanup(plt.close, viewer.fig)

        legend = viewer.ax.get_legend()
        self.assertIsNotNone(legend)
        labels = {text.get_text() for text in legend.get_texts()}
        self.assertIn("drones", labels)
        self.assertIn("boundary", labels)

    def test_sdqn_viewer_populates_rebuilt_main_axes_immediately(self) -> None:
        simulation = make_environment()
        self.addCleanup(simulation.close)
        viewer = SDQNViewer(simulation, background_type="none")
        self.addCleanup(plt.close, viewer.fig)

        np.testing.assert_array_equal(
            viewer.agents._drones_artist.get_xdata(),
            np.array([50.0]),
        )
        self.assertGreater(len(viewer.obstacles._artists), 0)


if __name__ == "__main__":
    unittest.main()
