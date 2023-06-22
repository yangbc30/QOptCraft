from pathlib import Path
import pickle

import numpy as np
from numpy.typing import NDArray, ArrayLike


BasisPhoton = list[tuple[int, ...]]


def get_photon_basis(modes: int, photons: int) -> BasisPhoton:
    """Return a basis for the Hilbert space with n photons and m modes.
    If the basis was saved retrieve it, otherwise the function creates
    and saves the basis to a file.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        BasisPhoton: basis of the Hilbert space.
    """
    folder = Path(f"save_basis/m={modes} n={photons}")
    folder.mkdir(parents=True, exist_ok=True)
    basis_path = folder / "photon.pkl"
    basis_path.touch()

    try:
        with basis_path.open("rb") as f:
            basis = pickle.load(f)

    except EOFError:
        basis = _photon_basis(modes, photons)
        with basis_path.open("wb") as f:
            pickle.dump(basis, f)
        print(f"Basis saved in {basis_path}.")

    return basis


def _photon_basis(modes: int, photons: int) -> BasisPhoton:
    """Given a number of photons and modes, generate the basis of the Hilbert space.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        BasisPhoton: basis of the Hilbert space.
    """
    if photons < 0:
        photons = 0
    if modes == 1:
        return [(photons,)]

    new_basis = []
    for n in range(photons + 1):
        basis = _photon_basis(modes - 1, photons - n)
        for vector in basis:
            new_basis.append((n, *vector))
    return new_basis


def state_in_basis(fock_list: BasisPhoton, amplitudes: ArrayLike, basis: BasisPhoton) -> NDArray:
    """Given a vector in terms of elements of a basis and amplitudes,
    output the state vector.

    Args:
        fock_list (BasisPhoton): basis fock states.
        amplitudes (ArrayLike): amplitude of each fock state.
        basis (BasisPhoton): basis of the Hilbert space.

    Returns:
        NDArray: state in the given basis.
    """
    state = np.zeros(len(basis), dtype=complex)

    for i, fock in enumerate(fock_list):
        for j, basis_vector in enumerate(basis):
            if fock == basis_vector:
                state[j] = amplitudes[i]

    if not np.isclose(np.sum(state * state.conj()), 1):
        raise ValueError("Probabilites don't add to 1. Check if vectors can be found in basis.")

    return state
