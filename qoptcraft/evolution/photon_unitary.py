from typing import Literal

import numpy as np
from numpy.typing import NDArray
import scipy as sp

from qoptcraft.basis import photon_basis, hilbert_dim
from qoptcraft.math import log_matrix
from .photon_hamiltonian import photon_hamiltonian
from .fock_evolution import fock_evolution_heisenberg, fock_evolution_permanent


def photon_unitary(
    scattering_matrix: NDArray,
    photons: int,
    method: Literal[
        "heisenberg", "hamiltonian", "permanent glynn", "permanent ryser"
    ] = "permanent glynn",
) -> NDArray:
    """Unitary matrix of a linear interferometer with a number of photons as input.

    Args:
        scattering_matrix (NDArray): scattering matrix of a linear optical interferometer.
        photons (int): number of photons.
        method (str): method to calculate the multiphoton unitary. Options are 'heisenberg',
            'hamiltonian', 'permanent glynn' or 'permanent ryser'. Default is 'permanent glynn'.

    Returns:
        NDArray: image of the scattering matrix through the photonic homomorphism.
    """
    if method.split()[0] == "permanent":
        return photon_unitary_permanent(scattering_matrix, photons, method=method.split()[1])
    if method == "heisenberg":
        return photon_unitary_heisenberg(scattering_matrix, photons)
    if method == "hamiltonian":
        return photon_unitary_hamiltonian(scattering_matrix, photons)
    raise ValueError(
        "Options for method are 'heisenberg', 'hamiltonian', 'permanent glynn' or 'permanent ryser'"
    )


def photon_unitary_heisenberg(scattering_matrix: NDArray, photons: int) -> NDArray:
    """Standard evolution given by quantum mechanics.

    Args:
        scattering_matrix (NDArray): scattering matrix of a linear optical interferometer.
        photons (int): number of photons.

    Returns:
        NDArray: image of the scattering matrix through the photonic homomorphism.
    """
    modes = scattering_matrix.shape[0]
    dim = hilbert_dim(modes, photons)
    unitary = np.zeros((dim, dim), dtype=np.complex64)
    photon_basis = photon_basis(modes, photons)

    for col, fock_in in enumerate(photon_basis):
        unitary[:, col] = fock_evolution_heisenberg(scattering_matrix, fock_in).state_in_basis()
    return unitary


def photon_unitary_hamiltonian(scattering_matrix: NDArray, photons: int) -> NDArray:
    """ "Unitary matrix of a linear interferometer with a number of photons as input.
    Calculated by evolving the hamiltonian of the interferometer.

    Args:
        scattering_matrix (NDArray): scattering matrix of a linear optical interferometer.
        photons (int): number of photons.

    Returns:
        NDArray: image of the scattering matrix through the photonic homomorphism.
    """
    S_hamiltonian = log_matrix(scattering_matrix, method="schur")  # ? Best method??
    U_hamiltonian = photon_hamiltonian(S_hamiltonian, photons)
    return sp.linalg.expm(U_hamiltonian)


def photon_unitary_permanent(
    scattering_matrix: NDArray, photons: int, method: str = "glynn"
) -> NDArray:
    """Unitary matrix of a linear interferometer with a number of photons as input.
    Calculated using permanents:

    <S|phi(U)|T> = Per(U_ST) / sqrt(s1! ...sm! t1! ... tm!)

    Args:
        scattering_matrix (NDArray): scattering matrix of a linear optical interferometer.
        photons (int): number of photons.
        method (str): method to calculate the permanent. Options are 'glynn' and 'ryser'.
            Defaults to 'glynn'.

    Returns:
        NDArray: image of the scattering matrix through the photonic homomorphism.
    """
    modes = scattering_matrix.shape[0]
    dim = hilbert_dim(modes, photons)
    unitary = np.empty((dim, dim), dtype=np.complex128)
    photon_basis = photon_basis(modes, photons)

    for col, fock_in in enumerate(photon_basis):
        unitary[:, col] = fock_evolution_permanent(scattering_matrix, fock_in, method, photon_basis)
    return unitary
