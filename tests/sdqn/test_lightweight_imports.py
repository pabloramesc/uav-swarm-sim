import subprocess
import sys
import unittest

from sim.sdqn import DQNConfig


class LightweightImportTests(unittest.TestCase):
    def test_root_and_agents_keep_optional_stacks_unloaded(self) -> None:
        code = """
import sys
import sim
import sim.agents
assert 'matplotlib' not in sys.modules
assert 'tensorflow' not in sys.modules
assert 'rasterio' not in sys.modules
assert not any(name.startswith('sim.network') for name in sys.modules)
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_environment_and_rewards_do_not_import_ml_stack(self) -> None:
        code = """
import sys
import sim.sdqn
from sim.sdqn import RewardManager, SDQNEnvironment
assert 'tensorflow' not in sys.modules
assert 'keras' not in sys.modules
assert 'dqn' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_none_model_path_is_rejected_before_ml_import(self) -> None:
        code = """
import sys
from sim.sdqn.dqn_wrapper import DQNWrapper
try:
    DQNWrapper((8, 8, 1), model_path=None, train_mode=False)
except ValueError:
    pass
else:
    raise AssertionError('expected ValueError')
assert 'tensorflow' not in sys.modules
assert 'keras' not in sys.modules
assert 'dqn' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_dqn_configuration_is_validated_without_ml_imports(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_memory"):
            DQNConfig(memory_size=10, min_memory=11)
        with self.assertRaisesRegex(ValueError, "epsilon_min"):
            DQNConfig(epsilon=0.1, epsilon_min=0.2)


if __name__ == "__main__":
    unittest.main()
