import unittest

import numpy as np

from sim.agents import Agent, AgentsManager, Drone, RegistryNeighborProvider
from sim.agents.dynamics import StaticDynamics
from sim.core import SimulationSnapshot, Simulator
from sim.mobility.position_controller import ControllerContext, PositionController


class CountingAgent(Agent):
    def __init__(self, agent_id: int, agent_type: str) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            dynamics=StaticDynamics(),
            environment=object(),
        )
        self.update_count = 0

    def update(self, dt: float = 0.01) -> None:
        self.state[0] += dt
        self.time += dt
        self.update_count += 1


class RecordingController(PositionController):
    def __init__(self) -> None:
        self.initial_context: ControllerContext | None = None
        self.update_context: ControllerContext | None = None

    def initialize(self, context: ControllerContext) -> None:
        self.initial_context = context

    def update(self, context: ControllerContext) -> np.ndarray:
        self.update_context = context
        return np.zeros(3)


class RecordingNetwork:
    def __init__(self) -> None:
        self.initialized_positions: dict[int, np.ndarray] | None = None
        self.updates: list[tuple[float, dict[int, np.ndarray]]] = []
        self.waits: list[tuple[float, float]] = []
        self.closed = False

    def initialize(self, positions: dict[int, np.ndarray]) -> None:
        self.initialized_positions = positions

    def update(self, time: float, positions: dict[int, np.ndarray]) -> None:
        self.updates.append((time, positions))

    def wait_until(self, target_time: float, timeout: float) -> None:
        self.waits.append((target_time, timeout))

    def close(self) -> None:
        self.closed = True


class SimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = AgentsManager()
        # Deliberately use a global order different from the state mapping order.
        self.drone = CountingAgent(10, "drone")
        self.gcs = CountingAgent(20, "gcs")
        self.user = CountingAgent(30, "user")
        for agent in (self.drone, self.gcs, self.user):
            self.manager.register_agent(agent)

    @staticmethod
    def _states() -> dict[str, np.ndarray]:
        return {
            "gcs": np.full((1, 6), 2.0),
            "drone": np.full((1, 6), 1.0),
            "user": np.full((1, 6), 3.0),
        }

    def test_reset_by_agent_type_and_step_return_snapshots(self) -> None:
        simulator = Simulator(object(), self.manager, dt=0.5)
        reset_snapshot = simulator.reset(self._states())

        self.assertIsInstance(reset_snapshot, SimulationSnapshot)
        self.assertIs(simulator.snapshot, reset_snapshot)
        self.assertEqual(simulator.time, 0.0)
        self.assertEqual(simulator.step_count, 0)
        np.testing.assert_array_equal(simulator.drone_states, np.full((1, 6), 1.0))
        np.testing.assert_array_equal(simulator.gcs_states, np.full((1, 6), 2.0))
        np.testing.assert_array_equal(simulator.user_states, np.full((1, 6), 3.0))

        step_snapshot = simulator.step()

        self.assertEqual(step_snapshot.time, 0.5)
        self.assertEqual(step_snapshot.step_count, 1)
        self.assertEqual(self.drone.update_count, 1)
        self.assertEqual(self.gcs.update_count, 1)
        self.assertEqual(self.user.update_count, 1)
        self.assertEqual(step_snapshot.drone_states[0, 0], 1.5)

        # The previous transition result remains detached.
        self.assertEqual(reset_snapshot.drone_states[0, 0], 1.0)

    def test_step_before_reset_fails_clearly(self) -> None:
        simulator = Simulator(object(), self.manager)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            simulator.step()

    def test_legacy_full_array_uses_global_registration_order(self) -> None:
        simulator = Simulator(object(), self.manager)
        simulator.reset(
            np.vstack(
                [
                    np.full(6, 10.0),
                    np.full(6, 20.0),
                    np.full(6, 30.0),
                ]
            )
        )

        self.assertEqual(self.drone.state[0], 10.0)
        self.assertEqual(self.gcs.state[0], 20.0)
        self.assertEqual(self.user.state[0], 30.0)

    def test_reset_validates_type_rows_before_mutating_agents(self) -> None:
        simulator = Simulator(object(), self.manager)
        invalid = self._states()
        invalid["drone"] = np.zeros((2, 6))

        with self.assertRaisesRegex(ValueError, "1 rows"):
            simulator.reset(invalid)
        with self.assertRaises(RuntimeError):
            _ = self.drone.state

    def test_network_lifecycle_and_context_manager(self) -> None:
        network = RecordingNetwork()
        with Simulator(object(), self.manager, network=network) as simulator:
            simulator.reset(self._states())
            simulator.step()
            simulator.sync()

        self.assertEqual(set(network.initialized_positions or {}), {10, 20, 30})
        self.assertEqual(len(network.updates), 1)
        self.assertEqual(network.waits, [(0.01, 0.1)])
        self.assertTrue(network.closed)

    def test_manager_owns_global_id_uniqueness(self) -> None:
        first = CountingAgent(7, "drone")
        duplicate = CountingAgent(7, "user")
        manager = AgentsManager()
        manager.register_agent(first)

        with self.assertRaisesRegex(ValueError, "already registered"):
            manager.register_agent(duplicate)
        self.assertEqual(manager.drones.size, 1)
        self.assertEqual(manager.users.size, 0)
        self.assertEqual(manager.size, 1)

    def test_manager_keeps_registry_views_in_sync_when_removing_agents(self) -> None:
        removed = self.manager.unregister_agent(self.drone.agent_id)

        self.assertIs(removed, self.drone)
        self.assertNotIn(self.drone.agent_id, self.manager.all_agents)
        self.assertNotIn(self.drone.agent_id, self.manager.drones)
        self.assertEqual(self.manager.size, 2)
        self.assertEqual(self.manager.drones.size, 0)
        self.assertFalse(hasattr(self.manager.drones, "unregister"))

    def test_manager_registry_mapping_is_structurally_read_only(self) -> None:
        with self.assertRaises(TypeError):
            self.manager.registries["drone"] = self.manager.users  # type: ignore[index]

    def test_drone_controller_receives_neighbors_during_reset(self) -> None:
        manager = AgentsManager()
        first_controller = RecordingController()
        second_controller = RecordingController()
        first = Drone(
            1,
            object(),
            first_controller,
            RegistryNeighborProvider(1, manager.drones, manager.users),
        )
        second = Drone(
            2,
            object(),
            second_controller,
            RegistryNeighborProvider(2, manager.drones, manager.users),
        )
        manager.register_agent(first)
        manager.register_agent(second)

        Simulator(object(), manager).reset(
            {
                "drone": np.array(
                    [
                        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                        [4.0, 5.0, 6.0, 0.0, 0.0, 0.0],
                    ]
                )
            }
        )

        self.assertIsNotNone(first_controller.initial_context)
        self.assertIsNotNone(second_controller.initial_context)
        np.testing.assert_array_equal(
            first_controller.initial_context.drone_positions[2],
            np.array([4.0, 5.0, 6.0]),
        )
        np.testing.assert_array_equal(
            second_controller.initial_context.drone_positions[1],
            np.array([1.0, 2.0, 3.0]),
        )

    def test_all_controllers_observe_the_same_pre_step_state(self) -> None:
        manager = AgentsManager()
        first_controller = RecordingController()
        second_controller = RecordingController()
        first = Drone(
            1,
            object(),
            first_controller,
            RegistryNeighborProvider(1, manager.drones, manager.users),
        )
        second = Drone(
            2,
            object(),
            second_controller,
            RegistryNeighborProvider(2, manager.drones, manager.users),
        )
        manager.register_agent(first)
        manager.register_agent(second)
        simulator = Simulator(object(), manager)
        simulator.reset(
            {
                "drone": np.array(
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                )
            }
        )

        simulator.step(dt=1.0)

        np.testing.assert_array_equal(
            first_controller.update_context.drone_positions[2],
            np.array([10.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(
            second_controller.update_context.drone_positions[1],
            np.array([0.0, 0.0, 0.0]),
        )


if __name__ == "__main__":
    unittest.main()
