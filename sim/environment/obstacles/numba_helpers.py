import numpy as np
from numba import njit, prange


@njit(cache=True)
def center_distances_numba(pos: np.ndarray, center: np.ndarray) -> np.ndarray:
    deltas = pos - center
    distances = np.sqrt(np.sum(deltas**2, axis=1))
    return distances


@njit(cache=True)
def center_distances_and_directions_numba(
    pos: np.ndarray, center: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    deltas = center - pos
    distances = np.sqrt(np.sum(deltas**2, axis=1))
    directions = np.zeros_like(deltas)
    non_zero = distances > 0.0
    directions[non_zero] = deltas[non_zero] / distances[non_zero, None]
    return distances, directions


@njit(cache=True)
def circle_closest_point_numba(
    pos: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    distances, directions = center_distances_and_directions_numba(pos, center)
    closest = center + directions * radius
    closest[distances == 0.0] = np.array([radius, 0])
    return closest


@njit(cache=True)
def is_inside_rectangle_numba(
    pos: np.ndarray, left: float, right: float, bottom: float, top: float
) -> np.ndarray:
    return (
        (pos[:, 0] >= left)
        & (pos[:, 0] <= right)
        & (pos[:, 1] >= bottom)
        & (pos[:, 1] <= top)
    )


@njit(cache=True, parallel=False)
def rectangle_closest_point_numba(
    pos: np.ndarray, left: float, right: float, bottom: float, top: float
) -> np.ndarray:
    closest = np.zeros_like(pos)
    closest[:, 0] = np.clip(pos[:, 0], left, right)
    closest[:, 1] = np.clip(pos[:, 1], bottom, top)

    is_inside = is_inside_rectangle_numba(pos, left, right, bottom, top)
    inside_indices = np.arange(closest.shape[0])[is_inside]

    # Determine if the position is inside the rectangle
    for idx in prange(inside_indices.shape[0]):
        i = inside_indices[idx]
        px, py = pos[i, 0], pos[i, 1]
        distances = np.array(
            [abs(px - left), abs(px - right), abs(py - bottom), abs(py - top)]
        )
        min_index = np.argmin(distances)

        edge_x = np.array([left, right, px, px])
        edge_y = np.array([py, py, bottom, top])

        closest[i, 0] = edge_x[min_index]
        closest[i, 1] = edge_y[min_index]

    return closest


@njit(cache=True)
def rectangle_distances_numba(
    pos: np.ndarray, left: float, right: float, bottom: float, top: float
) -> np.ndarray:
    closest = rectangle_closest_point_numba(pos, left, right, bottom, top)
    deltas = closest - pos
    distances = np.sqrt(np.sum(deltas**2, axis=1))
    return distances


@njit(cache=True)
def rectangle_distances_and_directions_numba(
    pos: np.ndarray, left: float, right: float, bottom: float, top: float
) -> tuple[np.ndarray, np.ndarray]:
    closest = rectangle_closest_point_numba(pos, left, right, bottom, top)
    deltas = closest - pos
    distances = np.sqrt(np.sum(deltas**2, axis=1))
    directions = np.zeros_like(deltas)
    non_zero = distances > 0.0
    directions[non_zero] = deltas[non_zero] / distances[non_zero, None]
    return distances, directions


@njit(cache=True)
def rectangle_external_distances_numba(
    pos: np.ndarray, left: float, right: float, bottom: float, top: float
) -> np.ndarray:
    clipped = np.clip(pos, a_min=(left, bottom), a_max=(right, top))
    deltas = clipped - pos
    distances = np.sqrt(np.sum(deltas**2, axis=1))
    return distances
