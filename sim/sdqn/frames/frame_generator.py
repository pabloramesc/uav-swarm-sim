"""
Frame generator for rendering multi-channel frames from scenario state.
Supports configurable factory for geometry and layers.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from sim.environment import Environment

from .geometry import FrameGeometry, FrameGeometryFactory
from .layers import FrameLayer, FrameLayerFactory
from .state import ScenarioState


class FrameGenerator:
    def __init__(
        self,
        geometry: FrameGeometry,
        layers: list[FrameLayer],
        label: str = "frame",
    ):
        self.geometry = geometry
        self.layers = layers
        self.label = label

        self.channels = len(self.layers)
        self.shape = (self.geometry.height, self.geometry.width, self.channels)

    def generate(
        self, state: ScenarioState, dtype: str = "uint8"
    ) -> NDArray[np.float32 | np.uint8]:
        """Generate the full multi-channel frame for a given scenario state.

        Args:
            state (ScenarioState): Current positions of agent, neighbors, and users.
            dtype (str): Output dtype, either "float32" (values 0–1) or "uint8" (values 0–255).

        Returns:
            np.ndarray: Multi-channel frame array of shape (H, W, C) with the requested dtype.

        Raises:
            ValueError: If an unsupported dtype is provided.
        """
        frame = np.zeros(self.shape, dtype=np.float32)
        for i, layer in enumerate(self.layers):
            frame[..., i] = layer.generate_frame(state)

        if dtype == "float32":
            return np.clip(frame, 0.0, 1.0)

        elif dtype == "uint8":
            scaled = np.clip(frame, 0.0, 1.0) * 255.0
            return scaled.astype(np.uint8)

        else:
            raise ValueError(f"Unsupported dtype: {dtype}")


@dataclass
class FrameGeneratorFactory:
    geometry_factory: FrameGeometryFactory
    layer_factories: list[FrameLayerFactory]
    label: str = "frame"

    def create(self, env: Optional[Environment] = None) -> FrameGenerator:
        geometry = self.geometry_factory.create()
        layers = [f.create(geometry, env) for f in self.layer_factories]
        return FrameGenerator(geometry, layers, label=self.label)

    @property
    def shape(self):
        geo = self.geometry_factory.create()
        return (*geo.shape, len(self.layer_factories))
