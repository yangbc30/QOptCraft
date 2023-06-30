"""Stable Gram-Schmidt algorithm.
"""
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from .mat_inner_product import mat_inner_product, mat_norm


def gram_schmidt(basis: list[spmatrix] | list[NDArray]) -> list[spmatrix] | list[NDArray]:
    """Gram-Schmidt algorithm to orthonormalize a basis.

    Note:
        It turns out that the normal Gram-Schmidt algorithm suffers from
        numerical instability: Round-off errors can accumulate and destroy
        orthogonality of the resulting vectors. We introduce the modified
        Gram-Schmidt procedure to help remedy this issue.

    Args:
        basis (list[spmatrix]): basis to orthonormalize.

    Returns:
        list[spmatrix]: orthonormalized basis.

    References:
        Algorithm can be found in https://www.math.uci.edu/~ttrogdon/105A/html/Lecture23.html
    """
    dim = len(basis)
    orth_basis = []

    for j in range(dim):
        orth_basis.append(basis[j] / mat_norm(basis[j]))
        for k in range(j + 1, dim):
            basis[k] = basis[k] - mat_inner_product(orth_basis[j], basis[k]) * orth_basis[j]
    return orth_basis
