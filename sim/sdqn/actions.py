from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class Action(IntEnum):
    NOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


_DIRECTIONS: NDArray[np.float64] = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [-1.0, 0.0],
        [1.0, 0.0],
    ],
    dtype=np.float64,
)
NUM_ACTIONS = len(Action)


def action_to_displacement(action: Action | int) -> NDArray[np.float64]:
    """Return the unit horizontal displacement for one discrete action."""

    try:
        return _DIRECTIONS[Action(action)].copy()
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unknown SDQN action: {action!r}.") from exc


def actions_to_displacements(
    actions: NDArray[np.integer] | list[int] | tuple[int, ...],
) -> NDArray[np.float64]:
    """Convert a one-dimensional action batch to unit XY displacements."""

    values = np.asarray(actions)
    if values.ndim != 1:
        raise ValueError("Actions must be a one-dimensional array.")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Actions must contain integers.")
    if np.any((values < int(Action.NOP)) | (values >= NUM_ACTIONS)):
        raise ValueError("Actions contain an unknown SDQN action.")
    return _DIRECTIONS[values.astype(np.intp, copy=False)].copy()
