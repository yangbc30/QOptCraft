from numpy.typing import NDArray


def adjoint(H: NDArray, U: NDArray) -> NDArray:
    """Compute the Heisenberg evolution of a hamiltonian.
    This is done via the adjoint action of the unitary group
    on the algebra of hermitian matrices.

    Args:
        H (NDArray): matrix of the hamiltonian.
        U (NDArray): matrix of the unitary.

    Returns:
        NDArray: evolved hamiltonian.
    """
    return U @ H @ U.conj().T
