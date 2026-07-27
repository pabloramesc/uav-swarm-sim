import unittest

import numpy as np

from sim.math.path_loss_model import rssi_to_signal_quality, signal_strength


class PathLossModelTests(unittest.TestCase):
    def test_no_transmitters_produce_no_signal(self) -> None:
        receivers = np.zeros((3, 3))

        rssi = signal_strength(np.zeros((0, 3)), receivers)

        self.assertTrue(np.isneginf(rssi).all())
        np.testing.assert_array_equal(rssi_to_signal_quality(rssi), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
