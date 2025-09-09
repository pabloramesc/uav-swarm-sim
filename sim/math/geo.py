"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

"""
Geographic to ENU conversion utilities.
"""

import numpy as np
from numpy.typing import ArrayLike

LATDEG2METERS = 111320


def geo2enu(geo: ArrayLike, home: ArrayLike) -> np.ndarray:
    """
    Converts geographic coordinates (latitude, longitude, altitude) to local
    ENU (East-North-Up) coordinates in meters, relative to a reference point.

    Parameters
    ----------
    geo : ArrayLike
        Geographic coordinates [latitude, longitude, altitude] in (deg, deg, m).
        Can be a (3,) array for a single point or an (N, 3) array for multiple points.
    home : ArrayLike
        Reference geographic coordinates [latitude, longitude, altitude] in (deg, deg, m).
        Must be a (3,) array.

    Returns
    -------
    np.ndarray
        Local ENU coordinates [E, N, U] in meters.
        Returns a (3,) array for a single point or an (N, 3) array for multiple points.
    """
    geo = np.asarray(geo, dtype=float)  # Ensure geo is a numpy array
    home = np.asarray(home, dtype=float)
    if home.shape != (3,):
        raise ValueError("Home must be a (3,) array.")

    geo_2d = np.atleast_2d(geo)            # Ensure geo is at least 2D (N, 3)
    
    enu = np.zeros_like(geo_2d)
    dlat = geo_2d[:, 0] - home[0]
    dlon = geo_2d[:, 1] - home[1]
    enu[:, 0] = dlon * LATDEG2METERS * np.cos(np.deg2rad(home[0]))  # East
    enu[:, 1] = dlat * LATDEG2METERS  # North
    enu[:, 2] = geo_2d[:, 2] - home[2]  # Up

    return enu.reshape(geo.shape)  # Return same shape as input


def enu2geo(enu: ArrayLike, home: ArrayLike) -> np.ndarray:
    """
    Converts local ENU (East-North-Up) coordinates in meters to geographic
    coordinates (latitude, longitude, altitude) relative to a reference point.

    Parameters
    ----------
    enu : ArrayLike
        Local ENU coordinates [E, N, U] in meters.
        Can be a (3,) array for a single point or an (N, 3) array for multiple points.
    home : ArrayLike
        Reference geographic coordinates [latitude, longitude, altitude] in (deg, deg, m).
        Must be a (3,) array.

    Returns
    -------
    np.ndarray
        Geographic coordinates [latitude, longitude, altitude] in (deg, deg, m).
        Returns a (3,) array for a single point or an (N, 3) array for multiple points.
    """
    enu = np.asarray(enu, dtype=float)  # Ensure enu is a numpy array
    home = np.asarray(home, dtype=float)
    if home.shape != (3,):
        raise ValueError("Home must be a (3,) array.")
    enu2d = np.atleast_2d(enu) # Ensure enu is at least 2D (N, 3)

    geo = np.zeros_like(enu2d)
    geo[:, 0] = home[0] + (enu2d[:, 1] / LATDEG2METERS)  # Latitude
    geo[:, 1] = home[1] + (
        enu2d[:, 0] / (LATDEG2METERS * np.cos(np.deg2rad(home[0])))
    )  # Longitude
    geo[:, 2] = home[2] + enu2d[:, 2]  # Altitude

    return geo.reshape(enu.shape)  # Return same shape as input

if __name__ == "__main__":
    home = np.array([37.7749, -122.4194, 30.0])  # Reference point (latitude, longitude, altitude)

    # Test single point conversion
    print("Single point conversion:")
    enu_point = np.array([100.0, 100.0, 10.0])
    print(f"- ENU coordinates: {enu_point}")

    # Convert ENU to geographic
    geo_point = enu2geo(enu_point, home)
    print(f"- Geographic coordinates: {geo_point}")

    # Convert geographic back to ENU
    enu_converted = geo2enu(geo_point, home)
    print(f"- Converted ENU coordinates: {enu_converted}")
    
    # Test multiple points conversion
    print("Multiple points conversion:")
    enu_points = np.array([
        [100.0, 100.0, 10.0],
        [-100.0, -100.0, 10.0],
        [100.0, -100.0, -10.0],        
    ])
    print(f"- ENU coordinates:\n{enu_points}")

    # Convert ENU to geographic
    geo_points = enu2geo(enu_points, home)
    print(f"- Geographic coordinates:\n{geo_points}")

    # Convert geographic back to ENU
    enu_converted = geo2enu(geo_points, home)
    print(f"- Converted ENU coordinates:\n{enu_converted}")