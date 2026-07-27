"""
Frame generator module with configurable geometry and layers.
"""

from .frame_generator import FrameGenerator, FrameGeneratorFactory
from .geometry import (
    FrameGeometry,
    FrameGeometryFactory,
    LogPolarGeometry,
    LogPolarGeometryFactory,
    SquareGeometry,
    SquareGeometryFactory,
)
from .layers import (
    FrameLayer,
    FrameLayerFactory,
    ObstaclesLayer,
    ObstaclesLayerFactory,
    SignalLayer,
    SignalLayerConfig,
    SignalLayerFactory,
)
from .state import (
    ScenarioState,
    get_agent_position,
    get_neighbor_positions,
    get_user_positions,
)

__all__ = [
    "FrameGenerator",
    "FrameGeneratorFactory",
    "FrameGeometry",
    "FrameGeometryFactory",
    "SquareGeometry",
    "SquareGeometryFactory",
    "LogPolarGeometry",
    "LogPolarGeometryFactory",
    "FrameLayer",
    "FrameLayerFactory",
    "ObstaclesLayer",
    "ObstaclesLayerFactory",
    "SignalLayer",
    "SignalLayerConfig",
    "SignalLayerFactory",
    "ScenarioState",
    "get_agent_position",
    "get_neighbor_positions",
    "get_user_positions",
]
