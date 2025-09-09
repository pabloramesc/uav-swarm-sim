import numpy as np


def cartesian_to_polar(xy: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    xy : np.ndarray
        Cartesian coordinates (x, y) in a 2D array of shape (N, 2).

    Returns
    -------
    np.ndarray
        Polar coordinates (r, theta) in a 2D array of shape (N, 2),
        where r is the radius and theta is the angle in radians.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.column_stack([r, theta])


def polar_to_cartesian(r_theta: np.ndarray) -> np.ndarray:
    """
    Convert polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    r_theta : np.ndarray
        Polar coordinates (r, theta) in a 2D array of shape (N, 2).

    Returns
    -------
    np.ndarray
        Cartesian coordinates (x, y) in a 2D array of shape (N, 2).
    """
    r = r_theta[:, 0]
    theta = r_theta[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate random 2D Cartesian coordinates
    np.random.seed(42)
    xy = np.random.uniform(-10, 10, size=(10, 2))

    # Convert to polar and back
    r_theta = cartesian_to_polar(xy)
    xy_reconstructed = polar_to_cartesian(r_theta)

    # Verify round-trip reconstruction is accurate
    if np.allclose(xy, xy_reconstructed):
        print("✅ Cartesian to polar and back conversion is accurate.")
    else:
        print("❌ Cartesian to polar and back conversion is NOT accurate.")

    # Generate spiral points for visualization
    theta = np.linspace(0, 10 * np.pi, 100)
    r = np.linspace(0, 10, 100)
    spiral_r_theta = np.column_stack([r, theta])
    spiral_xy = polar_to_cartesian(spiral_r_theta)

    # Cartesian coordinates
    plt.subplot(121)
    plt.plot(spiral_xy[:, 0], spiral_xy[:, 1], "b-")
    plt.title("Cartesian Coordinates")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")

    # Polar coordinates
    plt.subplot(122, projection="polar")
    plt.polar(spiral_r_theta[:, 1], spiral_r_theta[:, 0], "r-")
    plt.title("Polar Vectors (from origin)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
