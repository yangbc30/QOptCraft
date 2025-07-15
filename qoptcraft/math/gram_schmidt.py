"""Stable Gram-Schmidt algorithm."""

from collections.abc import Generator

from numpy.typing import NDArray
from numpy import isclose
from scipy.sparse import spmatrix

from .norms import hs_inner_product, hs_norm


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
        norm = hs_norm(basis[j])
        if not isclose(norm, 0):
            orth_basis.append(basis[j] / norm)
        for k in range(j + 1, dim):
            basis[k] = basis[k] - hs_inner_product(orth_basis[-1], basis[k]) * orth_basis[-1]
    return orth_basis


def gram_schmidt_generator(basis: list[spmatrix] | list[NDArray]) -> Generator:
    """Gram-Schmidt algorithm to orthonormalize a basis.

    Note:
        It turns out that the normal Gram-Schmidt algorithm suffers from
        numerical instability: Round-off errors can accumulate and destroy
        orthogonality of the resulting vectors. We introduce the modified
        Gram-Schmidt procedure to help remedy this issue.

    Args:
        basis (list[spmatrix]): basis to orthonormalize.

    Yields:
        NDArray or spmatrix: orthonormalized basis element.

    References:
        Algorithm can be found in https://www.math.uci.edu/~ttrogdon/105A/html/Lecture23.html
    """
    dim = len(basis)
    orth_basis = []

    for j in range(dim):
        orth_basis.append(basis[j] / hs_norm(basis[j]))
        yield orth_basis[j]
        for k in range(j + 1, dim):
            basis[k] = basis[k] - hs_inner_product(orth_basis[j], basis[k]) * orth_basis[j]
