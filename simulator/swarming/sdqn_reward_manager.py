import numpy as np
from simulator.environment import Environment
from simulator.math.distances import pairwise_self_distances


class SDQNRewardManager:

    def __init__(self, env: Environment, cell_size: float = 10.0) -> None:
        self.env = env
        self.cell_size = cell_size

        self.d_obs = 10.0
        self.d_ideal = 50.0
        self.d_max = 100.0
        self.max_links = 6

        self.w_dist = 1.0
        self.w_conn = 0.1
        self.w_expl = 0.0

        self.visited_cells: dict[tuple[int, int], float] = {}
        self.expire_time = 60.0

    def update(
        self, drone_positions: np.ndarray, time: float
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = pairwise_self_distances(drone_positions)
        distances[distances <= 0.0] = np.inf
        nearest_distances = np.min(distances, axis=-1)
        connected_neighbors = np.sum(distances < self.d_max, axis=-1)

        r_dist = self.distance_rewards(nearest_distances)
        r_conn = self.connectivity_rewards(connected_neighbors)
        r_expl = self.exploration_rewards(drone_positions, time)
        dones = self.is_collision(drone_positions)

        rewards = self.w_dist * r_dist + self.w_conn * r_conn + self.w_expl * r_expl
        # rewards = np.clip(rewards, -1.0, +1.0)
        rewards[dones] = -1.0

        return rewards, dones

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

    def distance_rewards(self, d: np.ndarray) -> np.ndarray:
        rewards = np.exp(-d**2/self.d_obs**2)
        return rewards

    def connectivity_rewards(self, connected_neighbors: np.ndarray) -> np.ndarray:
        rewards = satexp_links_reward(connected_neighbors, n_typ=1)
        return rewards

    def exploration_rewards(self, positions: np.ndarray, time: float) -> np.ndarray:
        rewards = np.zeros(positions.shape[0])
        for i, pos in enumerate(positions):
            cell = self.get_cell_indices(pos[0:2])

            if cell not in self.visited_cells:
                rewards[i] = +1.0
                self.visited_cells[cell] = time
                continue

            elapsed = time - self.visited_cells[cell]
            if elapsed > self.expire_time:
                rewards[i] = +1.0

            self.visited_cells[cell] = time

        return rewards

    def is_collision(self, positions: np.ndarray) -> np.ndarray:
        inside = self.env.is_inside(positions)
        collision = self.env.is_collision(positions, check_altitude=False)
        return ~inside | collision
    
    def _get_obstacles_distances(self, positions: np.ndarray) -> np.ndarray:
        obstacles = self.env.boundary_and_obstacles
        distances = np.zeros((len(obstacles), positions.shape[0]))
        for i, obs in enumerate(obstacles):
            distances[i] = obs.distance(positions)
        return np.min()

