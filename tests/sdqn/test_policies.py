import unittest

import numpy as np

from sim.sdqn.policies import SeededRandomPolicy


class SeededRandomPolicyTests(unittest.TestCase):
    def test_equal_seeds_produce_equal_actions(self) -> None:
        frames = np.zeros((4, 5, 5, 2), dtype=np.uint8)
        policies = [SeededRandomPolicy(seed=17) for _ in range(2)]

        np.testing.assert_array_equal(
            policies[0].act(frames),
            policies[1].act(frames),
        )

    def test_actions_match_batch_size_and_valid_range(self) -> None:
        frames = np.zeros((20, 3, 3, 1), dtype=np.uint8)
        actions = SeededRandomPolicy(seed=4, num_actions=3).act(frames)

        self.assertEqual(actions.shape, (20,))
        self.assertEqual(actions.dtype, np.int32)
        self.assertTrue(np.all((actions >= 0) & (actions < 3)))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            SeededRandomPolicy(num_actions=0)
        with self.assertRaisesRegex(ValueError, "shape"):
            SeededRandomPolicy().act(np.zeros((3, 3, 1), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
