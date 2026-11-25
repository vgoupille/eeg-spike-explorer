"""Type stubs for the Rust extension (PyO3)."""

import numpy as np
import numpy.typing as npt

def detect(
    x: npt.NDArray[np.float64], k: float = 8.0, smooth_n: int = 5, refractory_n: int = 50
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]: ...
