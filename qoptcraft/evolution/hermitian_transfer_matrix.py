import numpy as np
from ..evolution import photon_unitary
from ..math import generate_ggm_matrices, gram_schmidt
from ..operators import ObservableTensor
from ..basis import hilbert_dim


def get_HTM(photon_unitary: np.ndarray, H_basis: list[np.ndarray]):
    size = len(H_basis)
    HTM = np.empty((size, size), np.complex128)
    for j, basis_element in enumerate(H_basis):
        transfered_state = photon_unitary @ basis_element @ photon_unitary.conj().T
        for i, projector in enumerate(H_basis):
            HTM[i, j] = np.trace(projector @ transfered_state)

    return HTM


def get_H_basis(modes: int, photons: int, h_basis: list[np.ndarray] | None = None):
    if h_basis == None:
        h_basis = generate_ggm_matrices(modes, True)

    dim = hilbert_dim(modes, photons)
    identity = np.eye(dim) / np.sqrt(dim)
    H_basis = [identity]

    for order in range(1, photons + 1):
        new_basis = ObservableTensor(h_basis, order).to_orthnorm_basis(photons)
        len1 = len(H_basis)
        H_basis = gram_schmidt(H_basis + new_basis)
        len2 = len(H_basis)
        assert (
            len2 - len1 == hilbert_dim(modes, order) ** 2 - hilbert_dim(modes, order - 1) ** 2
        ), f"Error in generating basis: {len2 - len1} != {hilbert_dim(modes, order) ** 2 - hilbert_dim(modes, order - 1) ** 2}"

    return H_basis


def get_full_HTM(scatter_mat: np.ndarray, photons: int, h_basis: list[np.ndarray] | None = None):
    unitary = photon_unitary(scatter_mat, photons, "permanent glynn")
    modes = scatter_mat.shape[0]
    H_basis = get_H_basis(modes, photons, h_basis)
    full_HTM = get_HTM(unitary, H_basis)

    assert is_block_unitary_HTM(full_HTM, modes, photons) == True

    return full_HTM


def analyze_JS_image_space(modes: int, photons: int, order: int, h_basis: list[np.ndarray] | None = None):
    if h_basis == None:
        h_basis = generate_ggm_matrices(modes, True)

    H_basis = get_H_basis(modes, photons, h_basis)

    norm_obs = ObservableTensor(h_basis, order).to_orthnorm_basis(photons)

    basis_index = []
    for obs in norm_obs:
        for i, basis in enumerate(H_basis):
            if not np.isclose(np.trace(basis @ obs), 0):
                basis_index.append(i)

    basis_index = np.unique(basis_index)
    print(len(basis_index), len(norm_obs))
    assert len(basis_index) == len(norm_obs), "Basis elements do not match the orthonormal basis"
    print(f"Basis elements indices: {basis_index}")


def is_block_unitary_HTM(HTM: np.ndarray, modes: int, photons: int, rtol=1e-10, atol=1e-10):
    """
    Check if an n-photon m-mode HTM is a block unitary matrix

    Parameters:
        HTM: Hermitian transfer matrix to check
        modes: Modes of linear optical network
        photons: Photon number of multiphoton multimode state
        atol: Absolute tolerance for numerical comparisons

    Returns:
        tuple: bool - is_block_unitary
    """
    dim = hilbert_dim(modes, photons)
    expected_size = dim**2
    if HTM.shape != (expected_size, expected_size):
        return False, f"Dimension error: expected {expected_size}×{expected_size}"

    # Block dimensions and starting positions
    block_sizes = [
        hilbert_dim(modes, order) ** 2 - hilbert_dim(modes, order - 1) ** 2
        for order in range(photons + 1)
    ]
    block_starts = np.cumsum([0] + block_sizes[:-1])
    block_num = len(block_sizes)

    # Check if each diagonal block is unitary
    for i, (start, size) in enumerate(zip(block_starts, block_sizes)):
        block = HTM[start : start + size, start : start + size]

        # Use numpy to check unitarity: U @ U.H ≈ I
        product = block @ block.conj().T
        identity = np.eye(size, dtype=complex)

        if not np.allclose(product, identity, rtol=rtol, atol=atol):
            return False, f"Block {i+1} is not unitary"

    # Check if off-diagonal blocks are zero
    for i in range(block_num):
        for j in range(block_num):
            if i != j:
                row_slice = slice(block_starts[i], block_starts[i] + block_sizes[i])
                col_slice = slice(block_starts[j], block_starts[j] + block_sizes[j])

                if not np.allclose(HTM[row_slice, col_slice], 0, rtol=rtol, atol=atol):
                    return False, f"Off-diagonal block ({i+1},{j+1}) is not zero"

    return True
