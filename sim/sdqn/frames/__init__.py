"""
Frame generator module with configurable geometry and layers.
"""

from .frame_generator import FrameGenerator, FrameGeneratorFactory
from .geometry import SquareGeometry, SquareGeometryFactory
from .layers import (
    ObstaclesLayer,
    ObstaclesLayerFactory,
    SignalLayer,
    SignalLayerConfig,
    SignalLayerFactory
)
from .state import ScenarioState, get_neighbor_positions, get_user_positions, get_agent_position
