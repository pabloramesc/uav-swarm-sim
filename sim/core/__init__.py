"""Small, dependency-light building blocks for multi-agent simulations."""

from .clock import SimulationClock
from .network import NetworkBackend
from .simulator import Simulator
from .snapshot import SimulationSnapshot

__all__ = [
    "Simulator",
    "SimulationClock",
    "SimulationSnapshot",
    "NetworkBackend",
]
