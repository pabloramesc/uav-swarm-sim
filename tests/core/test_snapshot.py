import unittest

import numpy as np

from sim.core import SimulationSnapshot


class SimulationSnapshotTests(unittest.TestCase):
    def test_snapshot_detaches_and_protects_state_arrays(self) -> None:
        source = np.arange(12.0).reshape(2, 6)
        snapshot = SimulationSnapshot(
            time=1.5,
            step_count=3,
            states={"drone": source},
        )

        source[0, 0] = -1.0
        self.assertEqual(snapshot.drone_states[0, 0], 0.0)
        self.assertEqual(snapshot.gcs_states.shape, (0, 6))
        self.assertEqual(snapshot.user_states.shape, (0, 6))
        with self.assertRaises(ValueError):
            snapshot.drone_states[0, 0] = 99.0
        with self.assertRaises(TypeError):
            snapshot.states["drone"] = source

    def test_snapshot_metadata_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            SimulationSnapshot(-1.0, 0, {})
        with self.assertRaises(ValueError):
            SimulationSnapshot(0.0, -1, {})


if __name__ == "__main__":
    unittest.main()
