import pickle

from qoptcraft import config


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
    folder_path = config.SAVE_DATA_PATH / f"m={modes} n={photons}"
    folder_path.mkdir(parents=True, exist_ok=True)
    basis_path = folder_path / "photon.pkl"
    basis_path.touch()

    try:
        with basis_path.open("rb") as f:
            basis = pickle.load(f)

    except EOFError:
        basis = photon_basis(modes, photons)
        with basis_path.open("wb") as f:
            pickle.dump(basis, f)
        print(f"Photon basis saved in {basis_path}.")

    return basis


def photon_basis(modes: int, photons: int) -> BasisPhoton:
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
    for n in range(photons, -1, -1):
        basis = photon_basis(modes - 1, photons - n)
        for vector in basis:
            new_basis.append((n, *vector))
    return new_basis
