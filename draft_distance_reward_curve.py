import numpy as np
from matplotlib import pyplot as plt


def distance_reward(d: np.ndarray, d_ideal: float, d_max: float) -> np.ndarray:
    reward = np.where(
        d <= d_ideal,
        2.0 * d / d_ideal - 1.0,
        np.clip(1.0 - (d - d_ideal) / (d_max - d_ideal), 0.0, 1.0),
    )
    return reward


d = np.linspace(0.0, 200.0, 1000)
reward = distance_reward(d, 50.0, 100.0)


plt.plot(d, reward)
plt.grid()
plt.show()
