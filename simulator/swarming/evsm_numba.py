"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def calculate_links(position: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """
    Determines which neighbors are linked to the given position based on the EVSM algorithm.

    Parameters
    ----------
    position : np.ndarray
        A (2,) array with the position [x, y] of the agent in meters.
    neighbors : np.ndarray
        A (N, 2) array with [x, y] positions of N neighbors in meters.

    Returns
    -------
    np.ndarray
        A (N,) boolean array where True indicates that the corresponding neighbor is linked.
    """
    neighbors = neighbors.reshape(-1, 2)
    num_neighbors = neighbors.shape[0]
    links = np.zeros((num_neighbors,), dtype=np.bool_)
    for i in prange(num_neighbors):
        # Perform the acute angle test with neighbor1:
        # - If no other neighbors are inside the circle between agent and neighbor1, the angle is acute
        # - If the angle is acute, create a link to neighbor1
        neighbor1 = neighbors[i, :]
        center = (position + neighbor1) / 2  # Midpoint between agent and neighbor1
        radius = np.linalg.norm(position - neighbor1) / 2  #  Radius of the circle
        distance = np.inf
        for j in range(num_neighbors):
            if j == i:
                continue
            neighbor2 = neighbors[j, :]
            distance = np.linalg.norm(neighbor2 - center)
            if distance <= radius:  # If another neighbor is inside, skip
                break
        if distance > radius:  # If no neighbors are inside, create a link
            links[i] = True
    return links


@njit(cache=True)
def calculate_control_force(
    position: np.ndarray, linked_neighbors: np.ndarray, ln: float = 1.0, ks: float = 1.0
) -> np.ndarray:
    """
    Calculates the control force based on linked neighbors using a spring model.

    Parameters
    ----------
    position : np.ndarray
        A (2,) array representing the position [x, y] of the current agent in meters.
    linked_neighbors : np.ndarray
        A (M, 2) array with [x, y] positions of M linked neighbors in meters.
    ln : float, optional
        Natural length of the spring in meters (default is 1.0).
    ks : float, optional
        Spring constant (default is 1.0).

    Returns
    -------
    np.ndarray
        A (2,) array representing the control force [fx, fy] in m/s^2.
    """
    linked_neighbors = linked_neighbors.reshape(-1, 2)
    deltas = linked_neighbors - position
    distances = np.sqrt(np.sum(deltas**2, axis=1))[:, np.newaxis]
    directions = np.where(distances > 0.0, deltas / distances, np.zeros_like(deltas))
    spring_forces = (distances - ln) * directions
    control_force = ks * np.sum(spring_forces, axis=0)
    return control_force


@njit(cache=True)
def calculate_damping_force(velocity: np.ndarray, kd: float = 1.0) -> np.ndarray:
    """
    Calculates the damping force to reduce velocity.

    Parameters
    ----------
    velocity : np.ndarray
        A (2,) array representing the velocity [vx, vy] of the agent in m/s.
    kd : float, optional
        Damping coefficient (default is 1.0).

    Returns
    -------
    np.ndarray
        A (2,) array representing the damping force [fx, fy] in m/s^2.
    """
    damping_force = -kd * velocity
    return damping_force


@njit(cache=True)
def calculate_sweep_angle(
    position: np.ndarray, neighbors: np.ndarray
) -> tuple[float, float]:
    """
    Calculates the sweep angle for exploration based on neighbors' positions.

    The sweep angle is the angular range in which no neighbor is visible by the agent.

    Parameters
    ----------
    position : np.ndarray
        A (2,) array representing the position [x, y] of the current agent in meters.
    neighbors : np.ndarray
        A (N, 2) array with [x, y] positions of N neighbors in meters.

    Returns
    -------
    tuple[float, float]
        The start and end angles of the sweep (in radians).
    """
    neighbors = neighbors.reshape(-1, 2)
    num_neighbors = neighbors.shape[0]
    if num_neighbors == 0:
        return (0.0, 0.0)
    vectors = neighbors - position
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles.sort()
    for i in range(num_neighbors):
        angle1 = angles[i - 1]
        angle2 = angles[i]
        sweep_angle = (angle2 - angle1) % (2 * np.pi)
        if sweep_angle >= np.pi / 2:
            return (angle1, angle2)
    return (np.nan, np.nan)


@njit(cache=True)
def calculate_avoidance_force(
    region_distances: np.ndarray,
    region_directions: np.ndarray,
    d_min: float = 1.0,
    ks: float = 1.0,
) -> np.ndarray:
    """
    Calculates the avoidance force to maintain a safe distance from regions.

    Parameters
    ----------
    region_distances : np.ndarray
        A (N, 1) array with distances to N regions in meters.
    region_directions : np.ndarray
        A (N, 2) array with [dx, dy] direction vectors to N regions.
    d_min : float, optional
        Minimum safe distance in meters (default is 1.0).
    ks : float, optional
        Repulsion constant (default is 1.0).

    Returns
    -------
    np.ndarray
        A (2,) array representing the avoidance force [fx, fy] in m/s^2.
    """
    region_distances = region_distances.reshape(-1, 1)
    region_directions = region_directions.reshape(-1, 2)
    is_near = region_distances < d_min
    region_distances = np.maximum(region_distances, 1e-2)
    repulsion_forces = (d_min / region_distances) ** 2 * (-region_directions) * is_near
    avoidance_force = ks * np.sum(repulsion_forces, axis=0)
    return avoidance_force


@njit(cache=True)
def calculate_exploration_force(
    region_distances: np.ndarray,
    region_directions: np.ndarray,
    sweep_angle: tuple[float, float],
    ln: float = 1.0,
    ks: float = 1.0,
) -> np.ndarray:
    """
    Calculates the exploration force based on visible regions.

    Parameters
    ----------
    region_distances : np.ndarray
        A (N, 1) array with distances to N regions in meters.
    region_directions : np.ndarray
        A (N, 2) array with [dx, dy] direction vectors to N regions.
    sweep_angle : tuple[float, float]
        The start and end angles of the sweep (in radians).
    ln : float, optional
        Natural length of the spring in meters (default is 1.0).
    ks : float, optional
        Spring constant (default is 1.0).

    Returns
    -------
    np.ndarray
        A (2,) array representing the exploration force [fx, fy] in m/s^2.
    """
    region_distances = region_distances.reshape(-1, 1)
    region_directions = region_directions.reshape(-1, 2)
    num_regions = region_distances.shape[0]
    if num_regions == 0:
        return np.zeros(2)
    region_angles = np.arctan2(region_directions[:, 1], region_directions[:, 0])
    # Get visible obstacle by checking if direction angles are inside the sweep angle
    diff = (region_angles - sweep_angle[0]) % (2 * np.pi)
    sweep = (sweep_angle[1] - sweep_angle[0]) % (2 * np.pi)
    is_visible = ((diff >= 0) & (diff <= sweep))[:, np.newaxis]
    weights = region_distances / np.sqrt(np.sum(region_distances**2)) * is_visible
    # Calculate the exploration forces
    forces = weights * (region_distances - ln) * region_directions
    exploration_force = ks * np.sum(forces, axis=0)
    return exploration_force
