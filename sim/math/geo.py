"""Geographic to local East-North-Up coordinate conversions."""

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

    geo_2d = np.atleast_2d(geo)  # Ensure geo is at least 2D (N, 3)

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
    enu2d = np.atleast_2d(enu)  # Ensure enu is at least 2D (N, 3)

    geo = np.zeros_like(enu2d)
    geo[:, 0] = home[0] + (enu2d[:, 1] / LATDEG2METERS)  # Latitude
    geo[:, 1] = home[1] + (
        enu2d[:, 0] / (LATDEG2METERS * np.cos(np.deg2rad(home[0])))
    )  # Longitude
    geo[:, 2] = home[2] + enu2d[:, 2]  # Altitude

    return geo.reshape(enu.shape)  # Return same shape as input
