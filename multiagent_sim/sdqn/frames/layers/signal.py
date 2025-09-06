"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

"""
Layer implementations for frame generation, including signal and user layers.
Each layer transforms ScenarioState into a 2D frame given a geometry.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from multiagent_sim.environment import Environment
from multiagent_sim.math.path_loss_model import rssi_to_signal_quality, signal_strength

from ..geometry import FrameGeometry
from ..state import ScenarioState
from .base import FrameLayer, FrameLayerFactory

PositionsGetter = Callable[[ScenarioState], np.ndarray]


def get_neighbor_positions(state: ScenarioState) -> np.ndarray:
    return state.neighbor_positions


def get_user_positions(state: ScenarioState) -> np.ndarray:
    return state.user_positions


@dataclass
class RadioModelConfig:
    tx_power: float = 20.0
    rssi_min: float = -80.0
    rssi_max: float = -30.0
    freq_mhz: float = 2412.0
    path_loss_exp: float = 2.4


@dataclass
class SignalLayerConfig:
    positions_getter: PositionsGetter
    label: str = "signal"
    plot_rssi: bool = True
    plot_tx: bool = True
    radio: RadioModelConfig = field(default_factory=RadioModelConfig)


class SignalLayer(FrameLayer):
    def __init__(
        self,
        geometry: FrameGeometry,
        environment: Environment = None,
        config: SignalLayerConfig = None,
        **kwargs,
    ):
        self.config = config or SignalLayerConfig(**kwargs)
        super().__init__(geometry, environment, label=self.config.label)

    def build_frame(self, state: ScenarioState) -> np.ndarray:
        frame = np.zeros(self.geometry.shape, dtype=np.float32)
        positions = self.config.positions_getter(state)

        if positions.shape[0] == 0:
            return frame

        relative_positions = positions - state.agent_position

        if self.config.plot_rssi:
            rssi = signal_strength(
                tx_positions=relative_positions,
                rx_positions=self.geometry.flat_cell_positions,
                f=self.config.radio.freq_mhz,
                n=self.config.radio.path_loss_exp,
                tx_power=self.config.radio.tx_power,
            )
            frame = rssi_to_signal_quality(
                rssi, vmin=self.config.radio.rssi_min, vmax=self.config.radio.rssi_max
            ).reshape(self.geometry.shape)

        if self.config.plot_tx:
            self.set_frame_cells(frame, positions=relative_positions, value=1.0)

        return frame


@dataclass
class SignalLayerFactory(FrameLayerFactory):
    positions_getter: PositionsGetter
    label: str

    def create(self, geo: FrameGeometry, env: Environment = None):
        config = SignalLayerConfig(
            positions_getter=self.positions_getter,
            label=self.label,
        )
        return SignalLayer(geometry=geo, environment=env, config=config)
