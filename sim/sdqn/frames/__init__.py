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
    SignalLayerFactory,
    get_neighbor_positions,
    get_user_positions,
)
from .state import ScenarioState
