import numpy as np

from ..environment import Environment


class SurfaceRandomWalker:
    def __init__(
        self,
        env: Environment,
        min_speed: float = 1.0,
        max_speed: float = 3.0,
        climb_rate: float = 0.2,
        turning_rate: float = 0.3,
    ) -> None:
        self.env = env
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.climb_rate = climb_rate
        self.turning_rate = turning_rate

        self.repulsion_radius = 5.0
        self.repulsion_force = 1.0
        
        self.state: np.ndarray = None
        
    def initialize(self, state: np.ndarray) -> None:
        if state.shape != (6,):
            raise ValueError("State must be a numpy array with shape (6,)")
        self.state = np.copy(state)

    def step(self, dt) -> np.ndarray:
        pos = self.state[0:2]
        vel = self.state[3:5]

        # Random velocity change
        random_direction = np.random.uniform(-1, 1, size=2)
        random_direction /= np.linalg.norm(random_direction)
        target_vel = random_direction * np.random.uniform(0.0, self.max_speed)

        # Smooth turn
        vel = (1 - self.turning_rate) * vel + self.turning_rate * target_vel

        # Obstacle avoidance
        # vel += self._obstacle_avoidance(pos)

        # Speed limiting
        speed = np.linalg.norm(vel)
        if speed > self.max_speed:
            vel *= self.max_speed / speed

        # Update horizontal motion
        self.state[3:5] = vel
        self.state[0:2] += vel * dt

        # Altitude tracking (surface-following)
        current_z = self.state[2]
        target_z = self.env.get_elevation(self.state[0:2])
        climb = np.clip(target_z - current_z, -self.climb_rate, self.climb_rate)
        self.state[5] = climb
        self.state[2] += climb * dt
        
        return self.state.copy()

    def _obstacle_avoidance(self, position: np.ndarray) -> np.ndarray:
        """
        Repulsion force to avoid nearby obstacles.
        """
        force = np.zeros(2)
        for obs in self.env.obstacles:
            d = obs.distance(position)
            if d < self.repulsion_radius:
                dir_vec = obs.direction(position)
                force -= (
                    self.repulsion_force * (self.repulsion_radius - d) * dir_vec
                )  # linear decay repulsion
        return force
