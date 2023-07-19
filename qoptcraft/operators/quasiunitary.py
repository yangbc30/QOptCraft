import numpy as np
from numpy.typing import NDArray


def random_quasiunitary(shape: tuple, seed: int | None = None) -> NDArray:
    """Create a random quasiunitary matrix.

    Args:
        shape (tuple): dimensions of the matrix.
        seed (int, optional): seed to calculate the random unitary. Defaults to None.

    Returns:
        NDArray: matrix of the quasiunitary.
    """
    dim_1, dim_2 = shape[0], shape[1]
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
    quasi_u = rng.standard_normal(dim_1 * dim_2).reshape(dim_1, dim_2)
    quasi_u += 1j * rng.standard_normal(dim_1 * dim_2)
    return quasi_u / np.sqrt(2)
