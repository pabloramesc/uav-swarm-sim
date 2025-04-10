import numpy as np

from simulator.agents.agent import Agent


class User(Agent):

    def __init__(self, id: int, max_speed: float = 2.0, max_acc: float = 1.0):
        super().__init__(id, type="user")
        self.max_speed = max_speed  # Velocidad máxima
        self.max_acc = max_acc  # Aceleración máxima

    def update(self, dt: float = 0.01) -> None:
        super().update(dt)
        self.random_walk(dt)
        
    def random_walk(self, dt: float = 0.01) -> None:
        # Generar una aceleración aleatoria limitada
        random_acc = np.random.uniform(-1, 1, size=3)  # Vector aleatorio en 3D
        random_acc /= np.linalg.norm(random_acc)  # Normalizar dirección
        random_acc *= np.random.uniform(0.0, self.max_acc)  # Escalar por la aceleración máxima

        # Actualizar la velocidad con la aceleración
        self.state[3:6] += random_acc * dt

        # Limitar la velocidad a la velocidad máxima
        speed = np.linalg.norm(self.state[3:6])
        if speed > self.max_speed:
            self.state[3:6] *= self.max_speed / speed

        # Actualizar la posición con la velocidad
        self.state[0:3] += self.state[3:6] * dt

