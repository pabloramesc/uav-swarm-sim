import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import ImageGrid

from ..simulators.sdqn_simulator import SDQNSimulator
from .simple_viewer import BackgroundType, SimpleViewer


class SDQNViewer(SimpleViewer):

    def __init__(
        self,
        sim: SDQNSimulator,
        fps: float = 10.0,
        background_type: BackgroundType = "rssi",
    ):
        super().__init__(
            sim=sim.sim, min_fps=fps, max_fps=fps, background_type=background_type
        )
        self.sdqn = sim

        self.frame_axes: list[Axes] = []
        self.frame_images: list[AxesImage] = []

        self._create_axes()
        self._init_frame_images()
        self.fig.tight_layout()

    def initialize(self) -> None:
        # Reconfigure axes limits to adapt to environment changes
        self.limits = self._calculate_axis_limits(limits=None)
        self._create_axes()
        self._init_frame_images()
        self.fig.tight_layout()
        
        super().initialize()

    def update(self, force: bool = False):
        super().update(force)
        self._update_frame_images()

    def _create_axes(self):
        num_channels = self._get_frame_channels()
        total_axes = 1 + num_channels  # Main sim plot + frame channels

        # Rewrite parent figure with new axes (main + frames)
        self.fig.clear()
        self.fig.set_size_inches(4 * total_axes, 4)
        self.axes = [
            self.fig.add_subplot(1, total_axes, i + 1) for i in range(total_axes)
        ]

        # Rewrite parent main axis
        self.ax = self.axes[0]
        self._configure_axes()  # Need to be configured again
        self._create_plotters()  # Create plotters again with new axes
        self.agents.marked_drone = 0
        self.background.show_colorbar = False

        # Store frame axes
        self.frame_axes = self.axes[1:]

    def _init_frame_images(self) -> None:
        # Remove old images
        for ax in self.frame_axes:
            ax.clear()
        self.frame_images = []

        shape = self._get_frame_shape()
        frames = np.zeros(shape)
        labels = self._get_frame_labels()
        cmaps = ["gray", "viridis", "plasma", "magma", "cividis"]
        radius = self._get_frame_radius()

        for i, ax in enumerate(self.frame_axes):
            im = self._init_frame(
                frame=frames[..., i],
                ax=ax,
                cmap=cmaps[i % len(cmaps)],
                label=labels[i] if i < len(labels) else "Channel {i}",
                radius=radius,
            )
            self.frame_images.append(im)

    def _update_frame_images(self) -> None:
        frames = self._get_drone_frames()
        for i, im in enumerate(self.frame_images):
            im.set_data(frames[..., i] / 255.0)

    def _get_drone_frames(self, drone_idx: int = 0) -> np.ndarray:
        if self.sdqn.sdqn_brain.frames is None:
            raise RuntimeError("SDQN Brain not initialized.")
        return self.sdqn.sdqn_brain.frames[drone_idx]

    def _get_frame_shape(self, iface_idx: int = 0) -> tuple[int, ...]:
        iface = self.sdqn.sdqn_brain.ifaces[iface_idx]
        return iface.frame_generator.shape

    def _get_frame_channels(self, iface_idx: int = 0) -> int:
        iface = self.sdqn.sdqn_brain.ifaces[iface_idx]
        return iface.frame_generator.channels

    def _get_frame_labels(self, iface_idx: int = 0) -> list[str]:
        iface = self.sdqn.sdqn_brain.ifaces[iface_idx]
        return [layer.label for layer in iface.frame_generator.layers]

    def _get_frame_radius(self, iface_idx: int = 0) -> float | None:
        iface = self.sdqn.sdqn_brain.ifaces[iface_idx]
        radius = getattr(iface.frame_generator.geometry, "radius", None)
        return radius

    def _init_frame(
        self,
        frame: np.ndarray,
        ax: Axes,
        cmap: str | None = None,
        label: str | None = None,
        radius: float | None = None,
    ) -> AxesImage:
        im = ax.imshow(frame / 255.0, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)

        if label is not None:
            ax.set_title(label)

        # self.fig.colorbar(im, ax=ax, use_gridspec=False)

        if radius is not None:
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.grid(True)
            im.set_extent((-radius, +radius, -radius, +radius))
            im.axes.set_xlim((-radius, +radius))
            im.axes.set_ylim((-radius, +radius))

        return im
