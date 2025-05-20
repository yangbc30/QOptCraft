from qoptcraft.utils import saved_basis


BasisPhoton = list[tuple[int, ...]]


@saved_basis(file_name="photon_basis.pkl")
def photon_basis(modes: int, photons: int, cache: bool = True) -> BasisPhoton:
    """Given a number of photons and modes, generate the basis of the Hilbert space.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        BasisPhoton: basis of the Hilbert space.
    """
    _ = cache  # only used by the decorator @saved_basis

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
