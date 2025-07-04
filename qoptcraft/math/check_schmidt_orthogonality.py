"""
Schmidt Orthogonality Check for Matrix Lists

This module provides functions to check Schmidt orthogonality between matrices
in a list using the Hilbert-Schmidt inner product: <A,B> = tr(A† @ B).

Functions:
    check_schmidt_orthogonality: Check pairwise Schmidt orthogonality
    schmidt_orthogonality_matrix: Get orthogonality matrix for all pairs
    is_schmidt_orthogonal_set: Check if entire set is orthogonal

Author: QOPTCRAFT Extension
License: Compatible with QOPTCRAFT license
"""

import numpy as np
from typing import List, Union, Tuple
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from .matrix_type import Matrix
from .norms import hs_inner_product


def schmidt_orthogonality_matrix(matrices: List[Matrix], tol: float = 1e-12) -> NDArray:
    """
    Compute Schmidt orthogonality matrix for a list of matrices.
    
    For a list of n matrices [M₀, M₁, ..., Mₙ₋₁], returns an n×n matrix O where
    O[i,j] = tr(Mᵢ† @ Mⱼ) represents the Hilbert-Schmidt inner product.
    
    If all matrices are Schmidt orthogonal, this returns the identity matrix
    (assuming the matrices are also normalized).
    
    Args:
        matrices: List of matrices (numpy arrays or scipy sparse matrices)
        tol: Tolerance for numerical precision
        
    Returns:
        n×n matrix where entry (i,j) is the Schmidt inner product of matrices i and j
        
    Raises:
        ValueError: If matrix list is empty or matrices have incompatible shapes
        
    Example:
        >>> import numpy as np
        >>> from qoptcraft.math import schmidt_orthogonality_matrix
        >>> 
        >>> # Create some test matrices
        >>> m1 = np.array([[1, 0], [0, 0]])
        >>> m2 = np.array([[0, 0], [0, 1]]) 
        >>> m3 = np.array([[0, 1], [1, 0]])
        >>> 
        >>> ortho_matrix = schmidt_orthogonality_matrix([m1, m2, m3])
        >>> print("Orthogonality matrix:")
        >>> print(ortho_matrix)
        >>> 
        >>> # Check if set is orthogonal
        >>> is_orthogonal = np.allclose(ortho_matrix, np.eye(3), atol=1e-12)
        >>> print(f"Is orthogonal set: {is_orthogonal}")
    """
    if not matrices:
        raise ValueError("Matrix list cannot be empty")
    
    n = len(matrices)
    
    # Check that all matrices have the same shape
    first_shape = matrices[0].shape
    for i, matrix in enumerate(matrices):
        if matrix.shape != first_shape:
            raise ValueError(f"Matrix {i} has shape {matrix.shape}, expected {first_shape}")
    
    # Compute orthogonality matrix
    ortho_matrix = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        for j in range(n):
            # Compute Hilbert-Schmidt inner product: tr(Mᵢ† @ Mⱼ)
            inner_product = hs_inner_product(matrices[i], matrices[j])
            ortho_matrix[i, j] = inner_product
    
    return ortho_matrix


def check_schmidt_orthogonality(matrices: List[Matrix], tol: float = 1e-12) -> Tuple[bool, NDArray]:
    """
    Check if a list of matrices is Schmidt orthogonal and return detailed results.
    
    Two matrices A and B are Schmidt orthogonal if tr(A† @ B) = 0.
    A set of matrices is Schmidt orthogonal if every pair is orthogonal.
    
    Args:
        matrices: List of matrices to check
        tol: Tolerance for considering values as zero
        
    Returns:
        Tuple of (is_orthogonal, orthogonality_matrix) where:
        - is_orthogonal: True if all off-diagonal elements are ≤ tol
        - orthogonality_matrix: n×n matrix of inner products
        
    Example:
        >>> # Check Pauli matrices (should be orthogonal)
        >>> sigma_x = np.array([[0, 1], [1, 0]])
        >>> sigma_y = np.array([[0, -1j], [1j, 0]])
        >>> sigma_z = np.array([[1, 0], [0, -1]])
        >>> 
        >>> is_orthogonal, ortho_mat = check_schmidt_orthogonality([sigma_x, sigma_y, sigma_z])
        >>> print(f"Pauli matrices orthogonal: {is_orthogonal}")
    """
    ortho_matrix = schmidt_orthogonality_matrix(matrices, tol)
    
    # Check if off-diagonal elements are zero (within tolerance)
    n = len(matrices)
    is_orthogonal = True
    
    for i in range(n):
        for j in range(n):
            if i != j:  # Off-diagonal elements
                if abs(ortho_matrix[i, j]) > tol:
                    is_orthogonal = False
                    break
        if not is_orthogonal:
            break
    
    return is_orthogonal, ortho_matrix


def is_schmidt_orthogonal_set(matrices: List[Matrix], tol: float = 1e-12) -> bool:
    """
    Check if a set of matrices is Schmidt orthogonal (simplified interface).
    
    Args:
        matrices: List of matrices to check
        tol: Tolerance for considering values as zero
        
    Returns:
        True if all matrices are pairwise Schmidt orthogonal
        
    Example:
        >>> # Quick check without detailed output
        >>> matrices = [sigma_x, sigma_y, sigma_z]
        >>> is_orthogonal = is_schmidt_orthogonal_set(matrices)
        >>> print(f"Set is orthogonal: {is_orthogonal}")
    """
    is_orthogonal, _ = check_schmidt_orthogonality(matrices, tol)
    return is_orthogonal


def print_orthogonality_report(matrices: List[Matrix], 
                             matrix_names: List[str] = None, 
                             tol: float = 1e-12) -> None:
    """
    Print a detailed orthogonality report for a list of matrices.
    
    Args:
        matrices: List of matrices to analyze
        matrix_names: Optional names for the matrices (defaults to M0, M1, ...)
        tol: Tolerance for numerical comparisons
        
    Example:
        >>> matrices = [sigma_x, sigma_y, sigma_z]
        >>> names = ['σₓ', 'σᵧ', 'σᵤ']
        >>> print_orthogonality_report(matrices, names)
    """
    if matrix_names is None:
        matrix_names = [f"M{i}" for i in range(len(matrices))]
    
    if len(matrix_names) != len(matrices):
        raise ValueError("Number of matrix names must match number of matrices")
    
    is_orthogonal, ortho_matrix = check_schmidt_orthogonality(matrices, tol)
    n = len(matrices)
    
    print("Schmidt Orthogonality Analysis")
    print("=" * 50)
    print(f"Number of matrices: {n}")
    print(f"Matrix shape: {matrices[0].shape}")
    print(f"Tolerance: {tol}")
    print(f"Overall orthogonal: {is_orthogonal}")
    print()
    
    print("Orthogonality Matrix (Hilbert-Schmidt inner products):")
    print("Rows and columns correspond to matrices in order given")
    print()
    
    # Print header
    header = "      " + "".join(f"{name:>12}" for name in matrix_names)
    print(header)
    
    # Print matrix with row labels
    for i in range(n):
        row_str = f"{matrix_names[i]:>4}: "
        for j in range(n):
            value = ortho_matrix[i, j]
            if abs(value.imag) < tol:
                # Real value
                row_str += f"{value.real:>12.6f}"
            else:
                # Complex value
                row_str += f"{value:>12.6f}"
        print(row_str)
    
    print()
    
    # Check individual pairs
    non_orthogonal_pairs = []
    for i in range(n):
        for j in range(i + 1, n):  # Only check upper triangle
            if abs(ortho_matrix[i, j]) > tol:
                non_orthogonal_pairs.append((i, j, ortho_matrix[i, j]))
    
    if non_orthogonal_pairs:
        print("Non-orthogonal pairs:")
        for i, j, value in non_orthogonal_pairs:
            print(f"  {matrix_names[i]} ⊥ {matrix_names[j]}: {value:.6f} (should be 0)")
    else:
        print("All pairs are orthogonal within tolerance!")
    
    print()
    
    # Check if diagonal elements are 1 (normalized)
    diagonal_elements = [ortho_matrix[i, i] for i in range(n)]
    if all(abs(elem.real - 1.0) < tol and abs(elem.imag) < tol for elem in diagonal_elements):
        print("All matrices are normalized (diagonal elements = 1)")
    else:
        print("Matrix norms (diagonal elements):")
        for i, elem in enumerate(diagonal_elements):
            print(f"  ||{matrix_names[i]}||² = {elem:.6f}")


def validate_schmidt_orthogonality_examples():
    """
    Run validation examples to test the Schmidt orthogonality functions.
    """
    print("Schmidt Orthogonality Validation Examples")
    print("=" * 60)
    
    # Example 1: Pauli matrices (should be orthogonal)
    print("\nExample 1: Pauli Matrices")
    print("-" * 30)
    
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.array([[1, 0], [0, 1]], dtype=complex)
    
    pauli_matrices = [sigma_x, sigma_y, sigma_z]
    print_orthogonality_report(pauli_matrices, ['σₓ', 'σᵧ', 'σᵤ'])
    
    # Example 2: Include identity (should not be orthogonal to others)
    print("\nExample 2: Pauli Matrices + Identity")
    print("-" * 40)
    
    pauli_with_id = [identity, sigma_x, sigma_y, sigma_z]
    print_orthogonality_report(pauli_with_id, ['I', 'σₓ', 'σᵧ', 'σᵤ'])
    
    # Example 3: Custom orthogonal set
    print("\nExample 3: Custom Orthogonal Matrices")
    print("-" * 40)
    
    # Create orthogonal 2x2 matrices
    m1 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|
    m2 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|
    m3 = np.array([[0, 1], [0, 0]], dtype=complex)  # |0⟩⟨1|
    m4 = np.array([[0, 0], [1, 0]], dtype=complex)  # |1⟩⟨0|
    
    custom_matrices = [m1, m2, m3, m4]
    print_orthogonality_report(custom_matrices, ['|0⟩⟨0|', '|1⟩⟨1|', '|0⟩⟨1|', '|1⟩⟨0|'])
    
    # Example 4: Using QOPTCRAFT basis functions
    print("\nExample 4: QOPTCRAFT Image Algebra Basis")
    print("-" * 45)
    
    try:
        from qoptcraft.basis import image_algebra_basis
        from qoptcraft.math import gram_schmidt
        
        # Get image algebra basis for 2 modes, 1 photon
        modes, photons = 2, 1
        basis_matrices = image_algebra_basis(modes, photons, orthonormal=False)
        
        print(f"Testing {len(basis_matrices)} matrices from image algebra basis")
        print(f"Modes: {modes}, Photons: {photons}")
        
        # Convert to dense arrays for easier handling
        basis_dense = []
        for matrix in basis_matrices:
            if hasattr(matrix, 'toarray'):  # Sparse matrix
                basis_dense.append(matrix.toarray())
            else:
                basis_dense.append(matrix)
        
        is_orthogonal = is_schmidt_orthogonal_set(basis_dense, tol=1e-10)
        print(f"Non-orthonormal basis is orthogonal: {is_orthogonal}")
        
        # Test orthonormalized version
        basis_orthonormal = gram_schmidt(basis_matrices)
        basis_ortho_dense = []
        for matrix in basis_orthonormal:
            if hasattr(matrix, 'toarray'):  # Sparse matrix
                basis_ortho_dense.append(matrix.toarray())
            else:
                basis_ortho_dense.append(matrix)
        
        is_orthonormal = is_schmidt_orthogonal_set(basis_ortho_dense, tol=1e-10)
        print(f"Orthonormalized basis is orthogonal: {is_orthonormal}")
        
        # Show orthogonality matrix for orthonormalized basis
        if len(basis_ortho_dense) <= 6:  # Only show if not too large
            ortho_matrix = schmidt_orthogonality_matrix(basis_ortho_dense)
            print(f"\nOrthogonality matrix shape: {ortho_matrix.shape}")
            print("Should be close to identity matrix:")
            print(np.round(ortho_matrix, 6))
            
            identity_check = np.allclose(ortho_matrix, np.eye(len(basis_ortho_dense)), atol=1e-10)
            print(f"Is identity matrix: {identity_check}")
        
    except ImportError:
        print("QOPTCRAFT basis functions not available for this example")


if __name__ == "__main__":
    validate_schmidt_orthogonality_examples()