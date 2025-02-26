from numpy.typing import NDArray

def commutator(A: NDArray, B: NDArray) -> NDArray:
    """Return the commutator of two matrices A and B"""
    return A @ B - B @ A
