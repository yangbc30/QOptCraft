from numpy.typing import NDArray


def adjoint_evol(hamiltonian: NDArray, unitary: NDArray) -> NDArray:
    """Compute the Heisenberg evolution of a hamiltonian.
    This is done via the adjoint action of the unitary group
    on the algebra of hermitian matrices.

    Args:
        hamiltonian (NDArray): matrix of the hamiltonian.
        unitary (NDArray): matrix of the unitary.

    Returns:
        NDArray: evolved hamiltonian.
    """
    return unitary @ hamiltonian @ unitary.conj().T
