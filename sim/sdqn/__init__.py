"""Swarm DQN simulation, observations, rewards, and optional model support.

Exports are resolved lazily so importing :mod:`sim.sdqn` never imports
TensorFlow, Keras, or dqn-lab.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Action",
    "NUM_ACTIONS",
    "action_to_displacement",
    "actions_to_displacements",
    "RewardConfig",
    "RewardManager",
    "SDQNEnvironmentConfig",
    "SDQNEnvironment",
    "cartesian_frame_factory",
    "default_frame_factory",
    "SeededRandomPolicy",
    "SDQNSimulator",
    "DQNConfig",
    "DQNWrapper",
    "SDQNTrainer",
]

_EXPORTS = {
    "Action": (".actions", "Action"),
    "NUM_ACTIONS": (".actions", "NUM_ACTIONS"),
    "action_to_displacement": (".actions", "action_to_displacement"),
    "actions_to_displacements": (".actions", "actions_to_displacements"),
    "RewardConfig": (".rewards", "RewardConfig"),
    "RewardManager": (".rewards", "RewardManager"),
    "SDQNEnvironmentConfig": (".environment", "SDQNEnvironmentConfig"),
    "SDQNEnvironment": (".environment", "SDQNEnvironment"),
    "cartesian_frame_factory": (".environment", "cartesian_frame_factory"),
    "default_frame_factory": (".environment", "default_frame_factory"),
    "SeededRandomPolicy": (".policies", "SeededRandomPolicy"),
    "SDQNSimulator": (".simulator", "SDQNSimulator"),
    "DQNConfig": (".dqn_wrapper", "DQNConfig"),
    "DQNWrapper": (".dqn_wrapper", "DQNWrapper"),
    "SDQNTrainer": (".trainer", "SDQNTrainer"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
