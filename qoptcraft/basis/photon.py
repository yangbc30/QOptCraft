import pickle

from numba import jit

from qoptcraft import config


BasisPhoton = list[tuple[int, ...]]


def _saved_photon_basis(modes: int, photons: int) -> BasisPhoton:
    """Return a basis for the Hilbert space with n photons and m modes.
    If the basis was saved retrieve it, otherwise the function creates
    and saves the basis to a file.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        BasisPhoton: basis of the Hilbert space.
    """
    folder_path = config.SAVE_DATA_PATH / f"m={modes} n={photons}"
    folder_path.mkdir(parents=True, exist_ok=True)
    basis_path = folder_path / "photon.pkl"
    basis_path.touch()

    try:
        with basis_path.open("rb") as f:
            basis = pickle.load(f)

    except EOFError:
        basis = photon_basis(modes, photons, cache=False)
        with basis_path.open("wb") as f:
            pickle.dump(basis, f)
        print(f"Photon basis saved in {basis_path}.")

    return basis


def photon_basis(modes: int, photons: int, cache: bool = True) -> BasisPhoton:
    """Given a number of photons and modes, generate the basis of the Hilbert space.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        BasisPhoton: basis of the Hilbert space.
    """

    if cache:
        return _saved_photon_basis(modes, photons)

    if photons < 0:
        photons = 0
    if modes == 1:
        return [(photons,)]

    new_basis = []
    for n in range(photons, -1, -1):
        basis = photon_basis(modes - 1, photons - n, cache=False)
        for vector in basis:
            new_basis.append((n, *vector))
    return new_basis


def complete_photon_basis(first_states: BasisPhoton) -> BasisPhoton:
    """Given a number of photons and modes, generate the basis of the Hilbert space.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        BasisPhoton: basis of the Hilbert space.
    """
    photons = sum(first_states[0])
    modes = len(first_states[0])
    basis = photon_basis(modes, photons)
    first_states = [tuple(fock) for fock in first_states]
    return [fock for fock in basis if fock not in first_states]
