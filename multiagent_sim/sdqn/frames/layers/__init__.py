"""
Defines frame layers that generate single-channel data for simulation frames.
"""

from .base import FrameLayer, FrameLayerFactory
from .signal import (
    SignalLayer,
    SignalLayerConfig,
    SignalLayerFactory,
    get_neighbor_positions,
    get_user_positions,
)
