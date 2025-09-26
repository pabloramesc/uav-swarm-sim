"""
Defines frame layers that generate single-channel data for simulation frames.
"""

from .base import FrameLayer, FrameLayerFactory
from .obstacles import ObstaclesLayer, ObstaclesLayerFactory
from .signal import (
    SignalLayer,
    SignalLayerConfig,
    SignalLayerFactory,
)
