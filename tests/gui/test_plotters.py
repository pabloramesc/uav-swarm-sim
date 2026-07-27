from __future__ import annotations

import unittest
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from sim.gui.plotters import BackgroundPlot  # noqa: E402


class BackgroundPlotTests(unittest.TestCase):
    def test_static_background_reuses_its_image_and_colorbar(self) -> None:
        elevation_map = SimpleNamespace(
            elevation_data=np.array([[0.0, 10.0]]),
            fused_img=np.zeros((4, 4, 3), dtype=np.uint8),
        )
        simulator = SimpleNamespace(
            environment=SimpleNamespace(elevation_map=elevation_map)
        )
        figure, axis = plt.subplots()
        self.addCleanup(plt.close, figure)
        background = BackgroundPlot(
            axis,
            simulator,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            background_type="fused",
            show_colorbar=True,
        )

        background.plot()
        background.plot()

        self.assertEqual(len(axis.images), 1)
        self.assertEqual(len(figure.axes), 2)

    def test_satellite_background_requires_loaded_tiles(self) -> None:
        simulator = SimpleNamespace(
            environment=SimpleNamespace(
                elevation_map=SimpleNamespace(satellite_img=None)
            )
        )
        figure, axis = plt.subplots()
        self.addCleanup(plt.close, figure)
        background = BackgroundPlot(
            axis,
            simulator,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            background_type="satellite",
        )

        with self.assertRaisesRegex(RuntimeError, "fetch_satellite=True"):
            background.plot()


if __name__ == "__main__":
    unittest.main()
