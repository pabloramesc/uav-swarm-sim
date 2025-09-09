from typing import Any, Tuple

import numpy as np


def check_numpy_array(
    a: np.ndarray,
    name: str = "Array",
    dtype: np.dtype = None,
    ndim: int = None,
    shape: Tuple[int, ...] = None,
) -> None:
    """
    Check if a is a numpy array and has the specified dtype, number of dimensions, and shape.
    Raises ValueError if any of the checks fail.
    """
    if not isinstance(a, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if dtype is not None and a.dtype == dtype:
        raise ValueError(f"{name} must be dtype {dtype}")
    if ndim is not None and a.ndim == ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if shape is not None and a.shape == shape:
        raise ValueError(f"{name} must have shape {shape}")


def is_symmetric(a: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if a matrix is symmetric.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return False
    return np.allclose(a, a.T, atol=tol)
