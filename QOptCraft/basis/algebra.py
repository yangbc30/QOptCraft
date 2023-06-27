from pathlib import Path
import pickle
import warnings

from scipy.sparse import spmatrix, csr_matrix, lil_matrix

from .photon import get_photon_basis, BasisPhoton
from qoptcraft.operators import creation, annihilation


BasisAlgebra = list[spmatrix]

warnings.filterwarnings(
    "ignore",
    message=(
        "Changing the sparsity structure of a csr_matrix is expensive."
        " lil_matrix is more efficient."
    ),
)


def get_algebra_basis(
    modes: int, photons: int, folder_path: Path = None
) -> tuple[BasisAlgebra, BasisAlgebra]:
    """Return a basis for the Hilbert space with n photons and m modes.
    If the basis was saved retrieve it, otherwise the function creates
    and saves the basis to a file.

    Args:
        photons (int): number of photons.
        modes (int): number of modes.

    Returns:
        BasisAlgebra, BasisAlgebra: basis of the algebra and the image algebra.
    """
    if folder_path is None:
        folder_path = Path("save_basis")
    folder_path = folder_path / f"m={modes} n={photons}"
    folder_path.mkdir(parents=True, exist_ok=True)

    basis_path = folder_path / "algebra.pkl"
    basis_image_path = folder_path / "image_algebra.pkl"
    basis_path.touch()  # create file if it doesn't exist
    basis_image_path.touch()
    try:
        with basis_path.open("rb") as f:
            basis = pickle.load(f)
        with basis_image_path.open("rb") as f:
            basis_image = pickle.load(f)

    except EOFError:
        basis, basis_image = _algebra_basis(modes, photons)
        with basis_path.open("wb") as f:
            pickle.dump(basis, f)
        with basis_image_path.open("wb") as f:
            pickle.dump(basis_image, f)
        print(f"Basis saved in {folder_path}")

    return basis, basis_image


def _algebra_basis(modes: int, photons: int) -> tuple[BasisAlgebra, BasisAlgebra]:
    """Generate the basis for the algebra and image algebra."""
    basis = []
    basis_image = []
    photon_basis = get_photon_basis(modes, photons)

    for mode_1 in range(modes):
        for mode_2 in range(mode_1 + 1):
            matrix = sym_matrix(mode_1, mode_2, modes)
            basis.append(matrix)
            basis_image.append(image_sym_matrix(mode_1, mode_2, photon_basis))

    # Divide into two loops to separate symmetric from antisymmetric matrices
    for mode_1 in range(modes):
        for mode_2 in range(mode_1):
            matrix = antisym_matrix(mode_1, mode_2, modes)
            basis.append(matrix)
            basis_image.append(image_antisym_matrix(mode_1, mode_2, photon_basis))

    return basis, basis_image


def sym_matrix(mode_1: int, mode_2: int, dim: int) -> spmatrix:
    """Create the element of the algebra i/2(|j><k| + |k><j|)."""
    matrix = csr_matrix((dim, dim), dtype="complex64")
    matrix[mode_1, mode_2] = 0.5j
    matrix[mode_2, mode_1] += 0.5j
    return matrix


def antisym_matrix(mode_1: int, mode_2: int, dim: int) -> spmatrix:
    """Create the element of the algebra 1/2(|j><k| - |k><j|)."""
    matrix = csr_matrix((dim, dim), dtype="complex64")
    matrix[mode_1, mode_2] = 0.5
    matrix[mode_2, mode_1] = -0.5

    return matrix


def image_sym_matrix(mode_1: int, mode_2: int, photon_basis: BasisPhoton) -> spmatrix:
    """Image of the symmetric basis matrix by the lie algebra homomorphism."""
    dim = len(photon_basis)
    matrix = lil_matrix((dim, dim), dtype="complex64")  # * efficient format for loading data

    for col, fock_ in enumerate(photon_basis):
        if fock_[mode_1] != 0:
            fock, coef = annihilation(mode_1, fock_)
            fock, coef_ = creation(mode_2, fock)
            matrix[photon_basis.index(fock), col] = 0.5j * coef * coef_

        if fock_[mode_2] != 0:
            fock, coef = annihilation(mode_2, fock_)
            fock, coef_ = creation(mode_1, fock)
            matrix[photon_basis.index(fock), col] += 0.5j * coef * coef_

    return matrix.tocsr()


def image_antisym_matrix(mode_1: int, mode_2: int, photon_basis: BasisPhoton) -> spmatrix:
    """Image of the antisymmetric basis matrix by the lie algebra homomorphism."""
    dim = len(photon_basis)
    matrix = lil_matrix((dim, dim), dtype="complex64")

    for col, fock_ in enumerate(photon_basis):
        if fock_[mode_1] != 0:
            fock, coef = annihilation(mode_1, fock_)
            fock, coef_ = creation(mode_2, fock)
            matrix[photon_basis.index(fock), col] = -0.5 * coef * coef_

        if fock_[mode_2] != 0:
            fock, coef = annihilation(mode_2, fock_)
            fock, coef_ = creation(mode_1, fock)
            matrix[photon_basis.index(fock), col] += 0.5 * coef * coef_

    return matrix.tocsr()
