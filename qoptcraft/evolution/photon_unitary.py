from typing import Literal

import numpy as np
from numpy.typing import NDArray
import scipy as sp

from qoptcraft.basis import get_photon_basis, hilbert_dim
from qoptcraft.math import logm_3
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
        scattering_matrix (NDArray): _description_
        fock_in (Fock): _description_
        method (str): method to calculate the multiphoton unitary. Options are 'heisenberg',
            'hamiltonian', 'permanent glynn' or 'permanent ryser'. Default is 'permanent glynn'.

    Returns:
        PureState: _description_
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
        scattering_matrix (NDArray): _description_
        photons (int): number of photons.

    Returns:
        NDArray: _description_
    """
    modes = scattering_matrix.shape[0]
    dim = hilbert_dim(modes, photons)
    unitary = np.zeros((dim, dim), dtype=np.complex64)
    photon_basis = get_photon_basis(modes, photons)

    for col, fock_in in enumerate(photon_basis):
        unitary[:, col] = fock_evolution_heisenberg(scattering_matrix, fock_in).state_in_basis()
    return unitary


def photon_unitary_hamiltonian(scattering_matrix: NDArray, photons: int) -> NDArray:
    """Calulate the multiphoton unitary from the scattering matrix of the interferometer
    through the hamiltonian evolution. First, the hamiltonian is calculated with the logarithm;
    next, it is evolved with the algebra homomorphism, and finally we exponentiate the evolved
    hamiltonian to get the lifted unitary.

    Note:
        Previously known as evolution_3.

    Args:
        scattering_matrix (NDArray): _description_
        photons (int): number of photons.

    Returns:
        NDArray: _description_
    """
    S_hamiltonian = logm_3(scattering_matrix)
    U_hamiltonian = photon_hamiltonian(S_hamiltonian, photons)
    return sp.linalg.expm(U_hamiltonian)


def photon_unitary_permanent(
    scattering_matrix: NDArray, photons: int, method: str = "glynn"
) -> NDArray:
    """<S|phi(U)|T> = Per(U_ST) / sqrt(s1! ...sm! t1! ... tm!)"""
    modes = scattering_matrix.shape[0]
    dim = hilbert_dim(modes, photons)
    unitary = np.empty((dim, dim), dtype=np.complex128)
    photon_basis = get_photon_basis(modes, photons)

    for col, fock_in in enumerate(photon_basis):
        unitary[:, col] = fock_evolution_permanent(scattering_matrix, fock_in, method, photon_basis)
    return unitary
