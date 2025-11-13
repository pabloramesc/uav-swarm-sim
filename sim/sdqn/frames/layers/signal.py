"""
Layer implementations for frame generation, including signal and user layers.
Each layer transforms ScenarioState into a 2D frame given a geometry.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from sim.environment import Environment
from sim.math.coverage import covered_positions
from sim.math.path_loss_model import rssi_to_signal_quality, signal_strength

from ..geometry import FrameGeometry
from ..state import PositionsGetter, ScenarioState, get_dummy_position
from .base import FrameLayer, FrameLayerFactory


@dataclass
class SignalLayerConfig:
    coverage_mode: Literal["none", "rssi", "binary"] = "none"
    plot_tx_points: bool = False
    plot_rx_points: bool = False
    plot_center: bool = False

    tx_power: float = 20.0
    rssi_min: float = -80.0
    rssi_max: float = -30.0
    freq_mhz: float = 2412.0
    path_loss_exp: float = 2.4


class SignalLayer(FrameLayer):
    def __init__(
        self,
        geometry: FrameGeometry,
        tx_positions_getter: Optional[PositionsGetter] = None,
        rx_positions_getter: Optional[PositionsGetter] = None,
        config: Optional[SignalLayerConfig] = None,
        label: str = "signal layer",
        **kwargs,
    ):
        self.tx_positions_getter = tx_positions_getter or get_dummy_position
        self.rx_positions_getter = rx_positions_getter or get_dummy_position
        self.config = config or SignalLayerConfig(**kwargs)
        super().__init__(
            geometry, environment=None, label=label, plot_center=self.config.plot_center
        )

    def build_frame(self, state: ScenarioState) -> np.ndarray:
        frame = np.zeros(self.geometry.shape, dtype=np.float32)

        tx_positions = self.tx_positions_getter(state) - state.agent_position
        rx_positions = self.rx_positions_getter(state) - state.agent_position

        if tx_positions.shape[0] == 0 and rx_positions.shape[0] == 0:
            return frame

        if self.config.coverage_mode == "none":
            pass

        elif self.config.coverage_mode == "rssi":
            rssi = signal_strength(
                tx_positions=tx_positions,
                rx_positions=self.cell_ground_positions,
                f=self.config.freq_mhz,
                n=self.config.path_loss_exp,
                tx_power=self.config.tx_power,
            )
            frame = rssi_to_signal_quality(
                rssi, vmin=self.config.rssi_min, vmax=self.config.rssi_max
            ).reshape(self.geometry.shape)

        elif self.config.coverage_mode == "binary":
            covered_mask = covered_positions(
                tx_positions=tx_positions,
                rx_positions=self.cell_ground_positions,
                tx_power=self.config.tx_power,
                min_rssi=self.config.rssi_min,
                freq_mhz=self.config.freq_mhz,
                path_loss_exp=self.config.path_loss_exp,
            ).reshape(self.geometry.shape)
            frame[covered_mask] = 0.5

        else:
            raise ValueError(f"Invalid converage mode '{self.config.coverage_mode}'")

        if self.config.plot_tx_points:
            self.set_frame_cells(
                frame, positions=tx_positions[:, 0:2], value=1.0, clip=True
            )

        if self.config.plot_rx_points:            
            covered_mask = covered_positions(
                tx_positions=np.vstack([tx_positions, np.zeros(3)]),
                rx_positions=rx_positions,
                tx_power=self.config.tx_power,
                min_rssi=self.config.rssi_min,
                freq_mhz=self.config.freq_mhz,
                path_loss_exp=self.config.path_loss_exp,
            )
            self.set_frame_cells(
                frame, positions=rx_positions[covered_mask, 0:2], value=0.5, clip=True
            )
            self.set_frame_cells(
                frame, positions=rx_positions[~covered_mask, 0:2], value=1.0, clip=True
            )

        return frame


@dataclass
class SignalLayerFactory(FrameLayerFactory):
    tx_positions_getter: Optional[PositionsGetter] = None
    rx_positions_getter: Optional[PositionsGetter] = None

    coverage_mode: Literal["none", "rssi", "binary"] = "none"
    plot_tx_points: bool = True
    plot_rx_points: bool = True

    label: str = "signal layer"
    plot_center: bool = True

    def create(
        self, geo: FrameGeometry, env: Optional[Environment] = None
    ) -> FrameLayer:
        config = SignalLayerConfig(
            coverage_mode=self.coverage_mode,
            plot_tx_points=self.plot_tx_points,
            plot_rx_points=self.plot_rx_points,
            plot_center=self.plot_center,
        )
        return SignalLayer(
            geometry=geo,
            tx_positions_getter=self.tx_positions_getter,
            rx_positions_getter=self.rx_positions_getter,
            config=config,
            label=self.label,
        )
