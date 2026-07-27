from typing import Any, Protocol

import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from ..sdqn.frames import FrameGenerator, FrameLayer
from .plotters import BackgroundType
from .simple_viewer import SimpleViewer


class SDQNViewEnvironment(Protocol):
    clock: Any
    environment: Any
    drone_states: NDArray[np.float64]
    user_states: NDArray[np.float64]
    gcs_states: NDArray[np.float64]
    last_frames: NDArray[np.uint8] | None
    frame_generators: list[FrameGenerator]


class SDQNViewer(SimpleViewer):
    def __init__(
        self,
        environment: SDQNViewEnvironment,
        fps: float = 10.0,
        background_type: BackgroundType = "rssi",
    ):
        super().__init__(
            sim=environment,  # type: ignore[arg-type]
            min_fps=fps,
            max_fps=fps,
            background_type=background_type,
        )
        self.sdqn_environment = environment

        self.frame_axes: list[Axes] = []
        self.frame_images: list[AxesImage] = []

        self._create_axes()
        self._init_frame_images()
        SimpleViewer.reset(self)
        self.fig.tight_layout()

    def reset(self) -> None:
        # Reconfigure axes limits to adapt to environment changes
        self.limits = self._calculate_axis_limits(limits=None)
        self._create_axes()
        self._init_frame_images()
        self.fig.tight_layout()

        super().reset()

    def render(self, force: bool = False):
        self._update_frame_images()
        super().render(force)

    def _create_axes(self):
        frame_generator = self._get_frame_generator()
        total_axes = 1 + frame_generator.channels  # Main sim plot + frame channels

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

        frame_generator = self._get_frame_generator()
        frames = np.zeros(frame_generator.shape)
        cmaps = ["gray", "viridis", "plasma", "magma", "cividis"]

        for i, ax in enumerate(self.frame_axes):
            im = self._init_frame(
                frame=frames[..., i],
                ax=ax,
                cmap=cmaps[i % len(cmaps)],
                layer=frame_generator.layers[i],
            )
            self.frame_images.append(im)

    def _update_frame_images(self) -> None:
        frames = self._get_drone_frames()
        for i, im in enumerate(self.frame_images):
            im.set_data(frames[..., i] / 255.0)

    def _get_drone_frames(self, drone_idx: int = 0) -> np.ndarray:
        if self.sdqn_environment.last_frames is None:
            raise RuntimeError("SDQN frames not initialized.")
        return self.sdqn_environment.last_frames[drone_idx]

    def _get_frame_generator(self, drone_idx: int = 0) -> FrameGenerator:
        return self.sdqn_environment.frame_generators[drone_idx]

    def _init_frame(
        self,
        frame: np.ndarray,
        ax: Axes,
        cmap: str | None = None,
        layer: FrameLayer | None = None,
    ) -> AxesImage:
        im = ax.imshow(frame / 255.0, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)

        if layer is None:
            return im

        geom = layer.geometry

        ax.set_title(layer.label)
        ax.set_xlabel(geom.xlabel or "")
        ax.set_ylabel(geom.ylabel or "")

        xlim, ylim = geom.xlim, geom.ylim
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if xlim and ylim:
            im.set_extent((xlim[0], xlim[1], ylim[0], ylim[1]))

        xticks, yticks = geom.xticks, geom.yticks
        xtick_labels, ytick_labels = geom.xtick_labels, geom.ytick_labels
        if xticks:
            ax.set_xticks(xticks)
        if yticks:
            ax.set_yticks(yticks)
        if xticks and xtick_labels:
            ax.set_xticklabels(xtick_labels)
        if yticks and ytick_labels:
            ax.set_yticklabels(ytick_labels)

        ax.grid(True)
        ax.set_aspect("auto")

        # self.fig.colorbar(im, ax=ax, use_gridspec=False)

        return im
