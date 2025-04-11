"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import numpy as np

LATDEG2METERS = 111320


def geo2xyz(geo: np.ndarray, home: np.ndarray) -> np.ndarray:
    """
    Converts geographic coordinates (latitude, longitude, altitude) to local
    Cartesian coordinates (X, Y, Z) in meters, relative to a reference point.

    Parameters
    ----------
    geo : np.ndarray
        Geographic coordinates [latitude, longitude, altitude] in degrees and meters.
    home : np.ndarray
        Reference geographic coordinates [latitude, longitude, altitude] in degrees and meters.

    Returns
    -------
    np.ndarray
        Local Cartesian coordinates [X, Y, Z] in meters.
    """
    xyz = np.zeros(3)
    dlat = geo[0] - home[0]
    dlon = geo[1] - home[1]
    xyz[0] = dlon * LATDEG2METERS * np.cos(np.deg2rad(home[0]))
    xyz[1] = dlat * LATDEG2METERS
    xyz[2] = geo[2] - home[2]
    return xyz


def xyz2geo(xyz: np.ndarray, home: np.ndarray) -> np.ndarray:
    """
    Converts local Cartesian coordinates (X, Y, Z) in meters to geographic
    coordinates (latitude, longitude, altitude) relative to a reference point.

    Parameters
    ----------
    xyz : np.ndarray
        Local Cartesian coordinates [X, Y, Z] in meters.
    home : np.ndarray
        Reference geographic coordinates [latitude, longitude, altitude] in degrees and meters.

    Returns
    -------
    np.ndarray
        Geographic coordinates [latitude, longitude, altitude] in degrees and meters.
    """
    geo = np.zeros(3)
    geo[0] = home[0] + (xyz[1] / LATDEG2METERS)
    geo[1] = home[1] + (xyz[0] / (LATDEG2METERS * np.cos(np.deg2rad(home[0]))))
    geo[2] = home[2] + xyz[2]
    return geo
