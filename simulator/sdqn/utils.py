import numpy as np

from simulator.environment import Environment


def gaussian_decay(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(x**2) / sigma**2)


def saturated_exponential(x: np.ndarray, tau: float) -> np.ndarray:
    return 1.0 - np.exp(-x / tau)


def distances_to_obstacles(env: Environment, pos: np.ndarray) -> np.ndarray:
    obstacles = env.boundary_and_obstacles
    distances = np.zeros((len(obstacles), pos.shape[0]))
    for i, obs in enumerate(obstacles):
        distances[i] = obs.distance(pos)
    return np.min(distances, axis=0)


class VisitedCells:

    def __init__(self, cell_size: float = 1.0):
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int], float] = {}

    def get_cell_indices(self, pos: np.ndarray) -> tuple[int, int]:
        px, py = pos
        i = int(px // self.cell_size)
        j = int(py // self.cell_size)
        return (i, j)

    def get_cell_origin(self, cell: tuple[int, int]) -> np.ndarray:
        i, j = cell
        x0 = i * self.cell_size
        y0 = j * self.cell_size
        return np.array([x0, y0])

    def get_cell_time(self, pos: np.ndarray) -> float:
        cell = self.get_cell_indices(pos)
        return self.cells.get(cell, None)

    def set_cell_time(self, pos: np.ndarray, time: float) -> None:
        cell = self.get_cell_indices(pos)
        self.cells[cell] = time

    def is_new_or_expired(
        self, pos: np.ndarray, now: float, expire_time: float
    ) -> bool:
        last_time = self.get_cell_time(pos)
        if last_time is None or (now - last_time) > expire_time:
            self.set_cell_time(pos, now)
            return True
        return False

    def reset(self) -> None:
        self.cells.clear()

    def get_cells_time(self, positions: np.ndarray) -> np.ndarray:
        cell_indices = (positions // self.cell_size).astype(int)
        unique_cells, inverse_indices = np.unique(
            cell_indices, axis=0, return_inverse=True
        )
        times = np.array([self.cells.get(tuple(cell), 0.0) for cell in unique_cells])
        return times[inverse_indices]

    def set_cells_time(self, positions: np.ndarray, time: float) -> None:
        cell_indices = (positions // self.cell_size).astype(int)
        unique_cells = np.unique(cell_indices, axis=0)
        for cell in unique_cells:
            self.cells[tuple(cell)] = time
