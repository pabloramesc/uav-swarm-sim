import unittest
from unittest.mock import patch

from sim.core import SimulationClock


class SimulationClockTests(unittest.TestCase):
    def test_reset_and_tick_track_simulation_and_monotonic_time(self) -> None:
        with patch(
            "sim.core.clock.time.monotonic",
            side_effect=[100.0, 102.5],
        ):
            clock = SimulationClock(dt=0.25)
            clock.reset()
            used_dt = clock.tick()

            self.assertEqual(used_dt, 0.25)
            self.assertEqual(clock.time, 0.25)
            self.assertEqual(clock.step_count, 1)
            self.assertEqual(clock.real_time, 2.5)

    def test_tick_requires_reset(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reset"):
            SimulationClock().tick()

    def test_durations_are_validated(self) -> None:
        for invalid in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    SimulationClock(invalid)

        clock = SimulationClock()
        clock.reset()
        for invalid in (0.0, -1.0, float("inf"), float("nan"), True):
            with self.subTest(step=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    clock.tick(invalid)

    def test_sync_uses_elapsed_monotonic_time(self) -> None:
        with (
            patch(
                "sim.core.clock.time.monotonic",
                side_effect=[10.0, 10.25],
            ),
            patch("sim.core.clock.time.sleep") as sleep,
        ):
            clock = SimulationClock(dt=1.0, sync_tolerance=0.1)
            clock.reset()
            clock.tick()
            clock.sync()

        sleep.assert_called_once_with(0.75)


if __name__ == "__main__":
    unittest.main()
