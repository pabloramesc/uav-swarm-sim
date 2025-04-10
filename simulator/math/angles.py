import numpy as np
from numpy.typing import ArrayLike
from typing import Union

from dataclasses import dataclass


def vector_angle(v: ArrayLike) -> Union[float, np.ndarray]:
    """
    Compute the angle(s) of 2D vector(s) with respect to the x-axis.

    Parameters
    ----------
    v : array_like
        A 2D vector with shape (2,) or an array of 2D vectors with shape (N, 2).

    Returns
    -------
    float or ndarray
        The angle(s) in radians between each vector and the x-axis. Returns a float if input is a single vector.

    Raises
    ------
    ValueError
        If the input is not of shape (2,) or (N, 2).
    """
    v = np.asarray(v)
    if v.ndim == 1 and v.shape[0] == 2:
        return np.arctan2(v[1], v[0])
    elif v.ndim == 2 and v.shape[1] == 2:
        return np.arctan2(v[:, 1], v[:, 0])
    else:
        raise ValueError("Input must have shape (2,) or (N, 2)")


def normalize_angle_pi(angle: ArrayLike) -> Union[float, np.ndarray]:
    """
    Normalize angle(s) to the range [-π, π).

    Parameters
    ----------
    angle : array_like
        Angle or array of angles in radians.

    Returns
    -------
    float or ndarray
        Normalized angle(s) in radians.
    """
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def normalize_angle_2pi(angle: ArrayLike) -> Union[float, np.ndarray]:
    """
    Normalize angle(s) to the range [0, 2π).

    Parameters
    ----------
    angle : array_like
        Angle or array of angles in radians.

    Returns
    -------
    float or ndarray
        Normalized angle(s) in radians.
    """
    return np.asarray(angle) % (2 * np.pi)


def diff_angle_pi(angle1: ArrayLike, angle2: ArrayLike) -> Union[float, np.ndarray]:
    """
    Compute the signed minimal difference between two angles, normalized to [-π, π).

    Parameters
    ----------
    angle1 : array_like
        First angle(s) in radians.
    angle2 : array_like
        Second angle(s) in radians.

    Returns
    -------
    float or ndarray
        Signed angular difference(s) in radians.
    """
    return normalize_angle_pi(np.asarray(angle1) - np.asarray(angle2))


def diff_angle_2pi(angle1: ArrayLike, angle2: ArrayLike) -> Union[float, np.ndarray]:
    """
    Compute the unsigned difference between two angles, normalized to [0, 2π).

    Parameters
    ----------
    angle1 : array_like
        First angle(s) in radians.
    angle2 : array_like
        Second angle(s) in radians.

    Returns
    -------
    float or ndarray
        Unsigned angular difference(s) in radians.
    """
    return normalize_angle_2pi(np.asarray(angle1) - np.asarray(angle2))


def is_angle_between(
    angle: ArrayLike, angle1: float, angle2: float
) -> Union[bool, np.ndarray]:
    """
    Check whether an angle lies between two other angles, in the [0, 2π) range.

    Parameters
    ----------
    angle : array_like
        Angle(s) to check (in radians).
    angle1 : float
        Start angle of the sector (in radians).
    angle2 : float
        End angle of the sector (in radians).

    Returns
    -------
    bool or ndarray
        True if angle(s) lies between angle1 and angle2, going counterclockwise.

    Notes
    -----
    This function assumes all angles are normalized to [0, 2π). If not, it will
    normalize them internally.
    """
    diff1 = diff_angle_2pi(angle, angle1)
    sweep = diff_angle_2pi(angle2, angle1)
    return (diff1 >= 0) & (diff1 <= sweep)


@dataclass
class SweepAngle:
    start: float
    stop: float

    @property
    def sweep(self) -> float:
        if self.start == self.stop:
            return 2 * np.pi
        diff_angle_2pi(self.stop, self.start)

    def contains(self, angle: float) -> bool:
        if self.start == self.stop:
            return angle != self.start
        return is_angle_between(angle, self.start, self.stop)
    
    def to_tuple(self) -> tuple[float, float]:
        return (self.start, self.stop)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Test angle normalization
    angles = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
    plt.figure(figsize=(10, 5))
    plt.plot(angles, normalize_angle_pi(angles), label="Normalize to [-π, π]")
    plt.plot(angles, normalize_angle_2pi(angles), label="Normalize to [0, 2π]")
    plt.axhline(0, color="black", lw=0.5, ls="--")
    plt.axhline(2 * np.pi, color="red", lw=0.5, ls="--", label="2π")
    plt.axhline(np.pi, color="red", lw=0.5, ls="--", label="π")
    plt.axhline(-np.pi, color="blue", lw=0.5, ls="--", label="-π")
    plt.title("Angle Normalization")
    plt.xlabel("Input Angle (radians)")
    plt.ylabel("Normalized Angle (radians)")
    plt.legend()
    plt.grid()
    plt.show()

    # Test vector angle and difference calculation
    vectors = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([-1, 0]),
        np.array([0, -1]),
    ]
    print("\nVector Angle Tests:")
    for v1 in vectors:
        print(f"Vector {v1} angle: {np.degrees(vector_angle(v1))} degrees")
        for v2 in vectors:
            angle1 = vector_angle(v1)
            angle2 = vector_angle(v2)
            diff = diff_angle_pi(angle2, angle1)
            print(f"Angle from {v1} to {v2}: {np.degrees(diff)} degrees")

    # Test angle between
    print("\nAngle Between Tests:")
    for angle1 in np.arange(0.0, 360.0, 90.0):
        for angle2 in np.arange(0.0, 360.0, 90.0):
            print(f"Angle1: {angle1}, Angle2: {angle2}")
            for angle in np.arange(0.0, 360.0 + 45.0, 45.0):
                result = is_angle_between(*np.deg2rad([angle, angle1, angle2]))
                print(f" - Is {angle} between? {result}")
