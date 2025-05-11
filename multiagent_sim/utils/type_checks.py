from typing import Any, Tuple

import numpy as np


def check_numpy_array(
    a: np.ndarray,
    name: str = "Array",
    shape: Tuple[int, ...] = None,
    dtype: np.dtype = None,
) -> None:
    if not isinstance(a, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if shape is not None and a.shape == shape:
        raise ValueError(f"{name} must have shape {shape}")
    if dtype is not None and a.dtype == dtype:
        raise ValueError(f"{name} must be dtype {dtype}")

def is_symmetric(a: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if a matrix is symmetric.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return False
    return np.allclose(a, a.T, atol=tol)