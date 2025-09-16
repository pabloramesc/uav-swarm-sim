"""
Layer implementations for frame generation, including signal and user layers.
Each layer transforms ScenarioState into a 2D frame given a geometry.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from sim.environment import Environment
from sim.math.path_loss_model import rssi_to_signal_quality, signal_strength

from ..geometry import FrameGeometry
from ..state import ScenarioState
from .base import FrameLayer, FrameLayerFactory

PositionsGetter = Callable[[ScenarioState], np.ndarray]


def get_neighbor_positions(state: ScenarioState) -> np.ndarray:
    return state.neighbor_positions


def get_user_positions(state: ScenarioState) -> np.ndarray:
    return state.user_positions


@dataclass
class SignalLayerConfig:
    plot_rssi: bool = True
    plot_tx: bool = True
    tx_power: float = 20.0
    rssi_min: float = -80.0
    rssi_max: float = -30.0
    freq_mhz: float = 2412.0
    path_loss_exp: float = 2.4


class SignalLayer(FrameLayer):
    def __init__(
        self,
        geometry: FrameGeometry,
        positions_getter: PositionsGetter,
        config: Optional[SignalLayerConfig] = None,
        label: str = "signal",
        **kwargs
    ):
        self.positions_getter = positions_getter
        self.config = config or SignalLayerConfig(**kwargs)
        super().__init__(geometry, environment=None, label=label)

    def build_frame(self, state: ScenarioState) -> np.ndarray:
        frame = np.zeros(self.geometry.shape, dtype=np.float32)
        positions = self.positions_getter(state)

        if positions.shape[0] == 0:
            return frame

        relative_positions = positions - state.agent_position

        if self.config.plot_rssi:
            rssi = signal_strength(
                tx_positions=relative_positions,
                rx_positions=self.cell_ground_positions,
                f=self.config.freq_mhz,
                n=self.config.path_loss_exp,
                tx_power=self.config.tx_power,
            )
            frame = rssi_to_signal_quality(
                rssi, vmin=self.config.rssi_min, vmax=self.config.rssi_max
            ).reshape(self.geometry.shape)

        if self.config.plot_tx:
            self.set_frame_cells(frame, positions=relative_positions[:, 0:2], value=1.0)

        return frame


@dataclass
class SignalLayerFactory(FrameLayerFactory):
    positions_getter: PositionsGetter
    label: str
    plot_rssi: bool = True
    plot_tx: bool = True

    def create(
        self, geo: FrameGeometry, env: Optional[Environment] = None
    ) -> FrameLayer:
        config = SignalLayerConfig(plot_rssi=self.plot_rssi, plot_tx=self.plot_tx)
        return SignalLayer(
            geometry=geo,
            positions_getter=self.positions_getter,
            config=config,
            label=self.label,
        )
