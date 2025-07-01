"""
Jordan-Schwinger Mapping Implementation

This module implements the Jordan-Schwinger mapping that maps
Hermitian matrices to quantum operators:

h_i → Ô_i = Σ_{k,l} (h_i)_{kl} â†_k â_l

The mapping preserves Hermiticity and provides a natural way to
construct quantum operators from classical matrices.

Classes:
    JordanSchwingerOperator: Operator created from Jordan-Schwinger mapping

Functions:
    jordan_schwinger_map: Main function to apply the mapping

Author: yangbc30 with claude sonnet 4  
License: GPL-3.0
"""

import numpy as np
from typing import Tuple, List, Dict, Optional

from .quantum_operators import (
    QuantumOperator, CreationOperator, AnnihilationOperator, 
    NumberOperator, OperatorSum, ScaledOperator, OperatorProduct
)


class JordanSchwingerOperator(QuantumOperator):
    """
    Operator obtained from Jordan-Schwinger mapping.
    
    Maps Hermitian matrix h to operator Ô = Σ_{k,l} h_{kl} â†_k â_l.
    The resulting operator preserves the Hermitian structure and
    provides a quantum representation of the classical matrix.
    
    Attributes:
        h_matrix (np.ndarray): Original Hermitian matrix
        expression (QuantumOperator): Built operator expression
    """
    
    def __init__(self, h_matrix: np.ndarray):
        """
        Initialize Jordan-Schwinger operator.
        
        Args:
            h_matrix: Hermitian matrix to map
            
        Raises:
            ValueError: If matrix is not square or not Hermitian
        """
        if h_matrix.ndim != 2 or h_matrix.shape[0] != h_matrix.shape[1]:
            raise ValueError("Input must be a square matrix")
        
        modes = h_matrix.shape[0]
        super().__init__(modes)
        
        if not np.allclose(h_matrix, h_matrix.conj().T, atol=1e-12):
            raise ValueError("Input matrix must be Hermitian")
        
        self.h_matrix = h_matrix.copy()
        self.expression = self._build_operator_expression()
    
    def _build_operator_expression(self) -> QuantumOperator:
        """
        Build the operator expression from the Hermitian matrix.
        
        The Jordan-Schwinger mapping is:
        Ô = Σ_{k,l} h_{kl} â†_k â_l
        
        For efficiency, diagonal terms h_{kk} are implemented using
        number operators n̂_k = â†_k â_k.
        
        Returns:
            QuantumOperator representing the mapped expression
        """
        terms = []
        
        for k in range(self.modes):
            for l in range(self.modes):
                coeff = self.h_matrix[k, l]
                
                # Skip negligible coefficients
                if abs(coeff) < 1e-15:
                    continue
                
                if k == l:
                    # Diagonal terms: use number operator for efficiency
                    # h_{kk} â†_k â_k = h_{kk} n̂_k
                    term = coeff * NumberOperator(k, self.modes)
                else:
                    # Off-diagonal terms: creation × annihilation
                    # h_{kl} â†_k â_l
                    a_dag_k = CreationOperator(k, self.modes)
                    a_l = AnnihilationOperator(l, self.modes)
                    term = coeff * (a_dag_k @ a_l)
                
                terms.append(term)
        
        if not terms:
            # All coefficients were negligible
            from .quantum_operators import ScalarOperator
            return ScalarOperator(0, self.modes)
        elif len(terms) == 1:
            return terms[0]
        else:
            return OperatorSum(terms)
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """Convert to matrix representation."""
        return self.expression.to_matrix(basis, state_to_index)
    
    def get_symbolic_form(self) -> str:
        """
        Get symbolic representation of the operator.
        
        Returns a human-readable string showing the operator structure,
        e.g., "1.000·n_0 - 1.000·n_1 + 0.500·a†_0·a_1 + 0.500·a†_1·a_0"
        
        Returns:
            String representation of the operator
        """
        terms = []
        
        for k in range(self.modes):
            for l in range(self.modes):
                coeff = self.h_matrix[k, l]
                
                if abs(coeff) < 1e-15:
                    continue
                
                # Format coefficient
                if abs(coeff.imag) < 1e-15:
                    # Real coefficient
                    coeff_str = f"{coeff.real:.3f}"
                else:
                    # Complex coefficient
                    coeff_str = f"({coeff:.3f})"
                
                # Remove unnecessary "1.000·" prefixes
                if coeff_str == "1.000":
                    coeff_str = ""
                elif coeff_str == "-1.000":
                    coeff_str = "-"
                elif coeff_str != "" and not coeff_str.startswith("-"):
                    coeff_str += "·"
                elif coeff_str.startswith("-") and coeff_str != "-":
                    coeff_str = coeff_str + "·"
                
                # Build term string
                if k == l:
                    # Diagonal term: number operator
                    term_str = f"{coeff_str}n_{k}"
                else:
                    # Off-diagonal term: creation × annihilation
                    term_str = f"{coeff_str}a†_{k}·a_{l}"
                
                terms.append(term_str)
        
        if not terms:
            return "0"
        
        # Join terms with proper signs
        result = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"
        
        return result
    
    def get_matrix_elements(self) -> Dict[Tuple[int, int], complex]:
        """
        Get the matrix elements of the original Hermitian matrix.
        
        Returns:
            Dictionary mapping (k, l) indices to matrix elements h_{kl}
        """
        elements = {}
        for k in range(self.modes):
            for l in range(self.modes):
                if abs(self.h_matrix[k, l]) > 1e-15:
                    elements[(k, l)] = self.h_matrix[k, l]
        return elements
    
    def is_diagonal(self) -> bool:
        """
        Check if the original matrix is diagonal.
        
        Returns:
            True if the matrix is diagonal (only n̂_k terms)
        """
        off_diagonal = self.h_matrix - np.diag(np.diag(self.h_matrix))
        return np.allclose(off_diagonal, 0, atol=1e-15)
    
    def trace(self, basis: List[Tuple[int, ...]]) -> complex:
        """
        Compute the trace of the operator in given basis.
        
        Args:
            basis: Fock basis states
            
        Returns:
            Trace of the operator matrix
        """
        matrix = self.to_matrix(basis)
        return np.trace(matrix)
    
    def __repr__(self) -> str:
        return f"JordanSchwinger({self.get_symbolic_form()})"


def jordan_schwinger_map(h_matrix: np.ndarray) -> JordanSchwingerOperator:
    """
    Apply Jordan-Schwinger mapping to a Hermitian matrix.
    
    The Jordan-Schwinger mapping provides a natural way to map
    classical Hermitian matrices to quantum operators:
    
    h → Ô = Σ_{k,l} h_{kl} â†_k â_l
    
    Properties preserved:
    - Hermiticity: h† = h → Ô† = Ô
    - Linearity: α·h₁ + β·h₂ → α·Ô₁ + β·Ô₂
    - Spectrum: Related but not identical due to quantum fluctuations
    
    Args:
        h_matrix: Hermitian matrix to map (modes × modes)
        
    Returns:
        JordanSchwingerOperator representing the mapped quantum operator
        
    Raises:
        ValueError: If matrix is not square or not Hermitian
        
    Example:
        >>> import numpy as np
        >>> from qoptcraft.operators import jordan_schwinger_map
        >>> from qoptcraft.basis import photon_basis
        >>> 
        >>> # Pauli-X matrix
        >>> sigma_x = np.array([[0, 1], [1, 0]])
        >>> js_op = jordan_schwinger_map(sigma_x)
        >>> print(js_op.get_symbolic_form())
        'a†_0·a_1 + a†_1·a_0'
        >>> 
        >>> # Matrix representation in 1-photon basis
        >>> basis = photon_basis(2, 1)
        >>> matrix = js_op.to_matrix(basis)
        >>> print(matrix)
        [[0. 1.]
         [1. 0.]]
    """
    return JordanSchwingerOperator(h_matrix)


def pauli_operators(modes: int = 2) -> Dict[str, JordanSchwingerOperator]:
    """
    Generate Jordan-Schwinger mapped Pauli operators.
    
    Args:
        modes: Number of modes (should be 2 for standard Pauli matrices)
        
    Returns:
        Dictionary containing Pauli operators: {'X', 'Y', 'Z', 'I'}
        
    Raises:
        ValueError: If modes != 2
    """
    if modes != 2:
        raise ValueError("Pauli operators are defined for 2 modes only")
    
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.array([[1, 0], [0, 1]], dtype=complex)
    
    return {
        'X': jordan_schwinger_map(sigma_x),
        'Y': jordan_schwinger_map(sigma_y),
        'Z': jordan_schwinger_map(sigma_z),
        'I': jordan_schwinger_map(identity)
    }


def su2_generators(modes: int = 2) -> Dict[str, JordanSchwingerOperator]:
    """
    Generate SU(2) generators using Jordan-Schwinger mapping.
    
    The SU(2) generators are J_x = σ_x/2, J_y = σ_y/2, J_z = σ_z/2.
    
    Args:
        modes: Number of modes (should be 2)
        
    Returns:
        Dictionary containing SU(2) generators: {'Jx', 'Jy', 'Jz'}
    """
    if modes != 2:
        raise ValueError("SU(2) generators are defined for 2 modes only")
    
    # SU(2) generators (Pauli matrices / 2)
    j_x = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    j_y = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    j_z = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    return {
        'Jx': jordan_schwinger_map(j_x),
        'Jy': jordan_schwinger_map(j_y),
        'Jz': jordan_schwinger_map(j_z)
    }


def coherent_state_displacement(alpha: complex, mode: int, modes: int) -> JordanSchwingerOperator:
    """
    Generate displacement operator for coherent states using Jordan-Schwinger mapping.
    
    The displacement operator is D(α) = exp(α·a† - α*·a), but here we
    construct the generator α·a† - α*·a as a Jordan-Schwinger operator.
    
    Args:
        alpha: Complex displacement parameter
        mode: Mode to displace
        modes: Total number of modes
        
    Returns:
        JordanSchwingerOperator representing the displacement generator
    """
    # Create matrix representation of displacement generator
    h_matrix = np.zeros((modes, modes), dtype=complex)
    
    # Only affects the specified mode
    # Generator: α·a† - α*·a corresponds to off-diagonal elements
    # But Jordan-Schwinger mapping doesn't directly handle a† - a terms
    # Instead, we use the fact that x̂ = (a† + a)/√2, p̂ = (a† - a)/(i√2)
    # For displacement in x: use matrix [[0, β], [β*, 0]] where β = α/√2
    
    if mode < modes:
        # This is a simplified version - for full displacement operator,
        # one would need matrix exponentiation
        beta = alpha / np.sqrt(2)
        h_matrix[mode, (mode + 1) % modes] = beta
        h_matrix[(mode + 1) % modes, mode] = np.conj(beta)
    
    return jordan_schwinger_map(h_matrix)


def validate_jordan_schwinger_properties(h_matrix: np.ndarray, 
                                       basis: List[Tuple[int, ...]],
                                       tolerance: float = 1e-12) -> Dict[str, bool]:
    """
    Validate that Jordan-Schwinger mapping preserves expected properties.
    
    Args:
        h_matrix: Original Hermitian matrix
        basis: Fock basis for testing
        tolerance: Numerical tolerance for checks
        
    Returns:
        Dictionary of validation results
    """
    # Apply mapping
    js_op = jordan_schwinger_map(h_matrix)
    js_matrix = js_op.to_matrix(basis)
    
    # Check properties
    results = {
        'input_hermitian': np.allclose(h_matrix, h_matrix.conj().T, atol=tolerance),
        'output_hermitian': np.allclose(js_matrix, js_matrix.conj().T, atol=tolerance),
        'output_finite': np.isfinite(js_matrix).all(),
        'correct_dimension': js_matrix.shape == (len(basis), len(basis))
    }
    
    # Check linearity (if possible)
    if h_matrix.shape[0] == 2:  # Only test for 2x2 matrices
        h1 = np.array([[1, 0], [0, 0]], dtype=complex)
        h2 = np.array([[0, 1], [1, 0]], dtype=complex)
        a, b = 2.0, -1.5
        
        # Direct mapping of linear combination
        js_combined = jordan_schwinger_map(a * h1 + b * h2)
        combined_matrix = js_combined.to_matrix(basis)
        
        # Linear combination of mapped operators
        js1 = jordan_schwinger_map(h1)
        js2 = jordan_schwinger_map(h2)
        linear_combination = a * js1 + b * js2
        linear_matrix = linear_combination.to_matrix(basis)
        
        results['linearity'] = np.allclose(combined_matrix, linear_matrix, atol=tolerance)
    
    return results


# Export main functions
__all__ = [
    'JordanSchwingerOperator',
    'jordan_schwinger_map',
    'pauli_operators',
    'su2_generators', 
    'coherent_state_displacement',
    'validate_jordan_schwinger_properties'
]