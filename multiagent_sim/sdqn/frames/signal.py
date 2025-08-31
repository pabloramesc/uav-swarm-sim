import numpy as np
from .square import SquareFrame, SquareFrameFactory
from ...math.path_loss_model import signal_strength, rssi_to_signal_quality
from dataclasses import dataclass


class SignalFrame(SquareFrame):

    def __init__(
        self,
        num_cells: int = 64,
        frame_radius: float = 100.0,
        tx_dbm: float = 20.0,
        rssi_min: float = -80.0,
        rssi_max: float = -30.0,
        freq_mhz: float = 2412.0,
        path_loss_exp: float = 2.4,
        plot_tx: bool = True,
        plot_center: bool = True,
        label: str = "signal_frame",
    ):
        super().__init__(
            num_cells=num_cells,
            frame_radius=frame_radius,
            channels=1,
            plot_center=plot_center,
            label=label,
        )

        self.tx_dbm = tx_dbm
        self.rssi_min = rssi_min
        self.rssi_max = rssi_max
        self.freq_mhz = freq_mhz
        self.path_loss_exp = path_loss_exp
        self.plot_tx = plot_tx

        self.agent = np.zeros(2)
        self.tx_positions = np.zeros((0, 2))
        self.relative_positions = np.zeros((0, 2))

    def set_data(self, agent: np.ndarray, tx_positions: np.ndarray, **kwargs):
        self.agent = agent
        self.tx_positions = tx_positions
        self.relative_positions = tx_positions - agent

    def update_frame(self):
        if self.relative_positions.shape[0] == 0:
            self.frame[:] = 0.0

        else:
            rssi = signal_strength(
                tx_positions=self.relative_positions,
                rx_positions=self.flat_cell_positions,
                f=self.freq_mhz,
                n=self.path_loss_exp,
                tx_power=self.tx_dbm,
            )
            self.frame[:] = rssi_to_signal_quality(
                rssi, vmin=self.rssi_min, vmax=self.rssi_max
            ).reshape(self.shape)

        if self.plot_tx:
            self.set_cells(positions=self.relative_positions, value=1.0)

        super().update_frame()

    def generate(self, update=True):
        frame = super().generate(update)
        scaled = np.clip(frame, 0.0, 1.0) * 255.0
        return scaled.astype(np.uint8)


@dataclass
class SignalFrameFactory(SquareFrameFactory):
    num_cells: int = 64
    frame_radius: float = 100.0
    tx_dbm: float = 20.0
    rssi_min: float = -80.0
    rssi_max: float = -30.0
    freq_mhz: float = 2412.0
    path_loss_exp: float = 2.4
    plot_tx: bool = True
    plot_center: bool = True
    label: str = "signal_frame"

    def create(self):
        return SignalFrame(
            num_cells=self.num_cells,
            frame_radius=self.frame_radius,
            tx_dbm=self.tx_dbm,
            rssi_min=self.rssi_min,
            rssi_max=self.rssi_max,
            freq_mhz=self.freq_mhz,
            path_loss_exp=self.path_loss_exp,
            plot_tx=self.plot_tx,
            plot_center=self.plot_center,
            label=self.label,
        )
