import numpy as np
from matplotlib import pyplot as plt


def linear_distance_reward(d: np.ndarray, d_ideal: float, d_max: float) -> np.ndarray:
    reward = np.where(
        d <= d_ideal,
        2.0 * d / d_ideal - 1.0,
        np.clip(1.0 - (d - d_ideal) / (d_max - d_ideal), 0.0, 1.0),
    )
    return reward

def linear_links_reward(n: np.ndarray, n_max: int) -> np.ndarray:
    reward = np.clip(n/n_max, 0.0, 1.0)
    return reward

def gaussian_decay_reward(x: np.ndarray, sigma: float) -> np.ndarray:
    reward = np.exp(-x**2/sigma**2)
    return reward

def saturated_exponential_reward(x: np.ndarray, tau: float) -> np.ndarray:
    reward = 1.0 - np.exp(-x/tau)
    return reward


d = np.linspace(0.0, 200.0, 1000)
r1 = linear_distance_reward(d, 50.0, 100.0)
r2 = -gaussian_decay_reward(d, sigma=10.0)

plt.title("Distance reward curves")
plt.plot(d, r1, label="linear")
plt.plot(d, r2, label="invexp")
plt.legend()
plt.grid()
plt.show()

n = np.arange(10)
r1 = linear_links_reward(n, n_max=6)
r2 = saturated_exponential_reward(n, tau=3)

plt.title("Connection reward curves")
plt.plot(n, r1, label="linear")
plt.plot(n, r2, label="satexp")
plt.legend()
plt.grid()
plt.show()
