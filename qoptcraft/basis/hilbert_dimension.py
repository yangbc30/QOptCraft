from scipy.special import comb


def hilbert_dim(modes: int, photons: int) -> int:
    """Dimension of the Hilbert space with m modes and n photons.

    Args:
        modes (int): number of modes.
        photons (int): number of photons.

    Returns:
        int: dimension.
    """
    return int(comb(modes + photons - 1, photons))
