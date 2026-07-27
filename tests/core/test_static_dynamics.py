import unittest

import numpy as np

from sim.agents.dynamics import StaticDynamics


class StaticDynamicsTests(unittest.TestCase):
    def test_step_accepts_control_and_keeps_state_unchanged(self) -> None:
        dynamics = StaticDynamics()
        initial = np.arange(6.0)
        dynamics.state = initial.copy()

        result = dynamics.step(0.5, control=np.ones(3))

        self.assertIsNone(result)
        np.testing.assert_array_equal(dynamics.state, initial)

    def test_step_validates_control_shape(self) -> None:
        dynamics = StaticDynamics()
        dynamics.state = np.zeros(6)
        with self.assertRaises(ValueError):
            dynamics.step(0.1, control=np.zeros(2))

    def test_state_is_owned_and_normalized(self) -> None:
        dynamics = StaticDynamics()
        source = np.arange(6)

        dynamics.state = source
        source[0] = 99

        self.assertEqual(dynamics.state.dtype, np.float64)
        self.assertEqual(dynamics.state[0], 0.0)


if __name__ == "__main__":
    unittest.main()
