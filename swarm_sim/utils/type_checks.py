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
