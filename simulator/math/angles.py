import numpy as np


def vector_angle(v: np.ndarray) -> float:
    return np.arctan2(v[1], v[0])


def normalize_angle_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def normalize_angle_2pi(angle: float) -> float:
    return angle % (2 * np.pi)


def diff_angle_pi(angle1: float, angle2: float) -> float:
    return normalize_angle_pi(angle1 - angle2)


def diff_angle_2pi(angle1: float, angle2: float) -> float:
    return normalize_angle_2pi(angle1 - angle2)


def is_angle_between(angle: float, angle1: float, angle2: float) -> bool:
    diff1 = diff_angle_2pi(angle, angle1)
    sweep = diff_angle_2pi(angle2, angle1)
    return 0 <= diff1 <= sweep


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
