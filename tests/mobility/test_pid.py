import unittest

import numpy as np

from sim.mobility import PIDController


class PIDControllerTests(unittest.TestCase):
    def test_control_preserves_error_shape(self) -> None:
        controller = PIDController(kp=1.0, ki=0.1, kd=0.2)
        output = controller.control(
            np.array([1.0, -1.0, 0.5]),
            derivative=np.zeros(3),
        )

        self.assertEqual(output.shape, (3,))
        with self.assertRaisesRegex(ValueError, "error shape changed"):
            controller.control(1.0)

        controller.reset()
        self.assertEqual(controller.control(1.0).shape, ())

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dt and tau"):
            PIDController(kp=1.0, dt=0.0)
        with self.assertRaisesRegex(ValueError, "gains"):
            PIDController(kp=-1.0)


if __name__ == "__main__":
    unittest.main()
