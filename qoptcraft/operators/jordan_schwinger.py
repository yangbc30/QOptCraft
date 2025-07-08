"""
Jordan-Schwinger Mapping Implementation - Higher Order Extension

This module implements the Jordan-Schwinger mapping that maps
Hermitian matrices to quantum operators at arbitrary orders:

First order:  h_i → Ô_i = Σ_{k,l} (h_i)_{kl} â†_k â_l
Second order: h_i h_j → Ô_{ij} = Σ_{k,l,m,n} (h_i)_{kl} (h_j)_{mn} â†_k â†_m â_n â_l
Higher order: (h_1 h_2 ... h_n) → Ô_{1,2,...,n} = Σ_{indices} ∏_i (h_i)_{k_i,l_i} ∏_j â†_{k_j} ∏_j â_{l_j}

The mapping preserves Hermiticity and provides a natural way to
construct quantum operators from classical matrices. All operators
are built using the evolve-based framework for consistency.

Classes:
    JordanSchwingerOperator: Operator created from Jordan-Schwinger mapping

Functions:
    jordan_schwinger_map: Main function to apply the mapping at any order
    pauli_operators: Generate Pauli operators via Jordan-Schwinger mapping

Author: QOPTCRAFT Extension  
License: Compatible with QOPTCRAFT license
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from collections import defaultdict
import itertools

from .quantum_operators import (
    QuantumOperator, CreationOperator, AnnihilationOperator, 
    NumberOperator, OperatorSum, ScaledOperator, OperatorProduct
)


class JordanSchwingerOperator(QuantumOperator):
    """
    Operator obtained from Jordan-Schwinger mapping of arbitrary order.
    
    Maps a product of Hermitian matrices h_1 * h_2 * ... * h_n to operator:
    Ô = Σ_{k_i,l_i} ∏_i (h_i)_{k_i,l_i} ∏_j â†_{k_j} ∏_j â_{l_j}
    
    The operator ordering follows the convention:
    - All creation operators â† come first (left to right as k_1, k_2, ..., k_n)
    - All annihilation operators â come last (right to left as l_n, l_{n-1}, ..., l_1)
    
    This gives the normal-ordered form which is most natural for calculations.
    
    Attributes:
        h_matrices (List[np.ndarray]): List of Hermitian matrices
        order (int): Order of the mapping (number of matrices)
        expression (QuantumOperator): Built operator expression
    """
    
    def __init__(self, h_matrices: Union[np.ndarray, List[np.ndarray]]):
        """
        Initialize Jordan-Schwinger operator of arbitrary order.
        
        Args:
            h_matrices: Single Hermitian matrix (order 1) or list of matrices (higher order)
            
        Raises:
            ValueError: If matrices are not square, not Hermitian, or incompatible sizes
        """
        # Handle single matrix input
        if isinstance(h_matrices, np.ndarray):
            h_matrices = [h_matrices]
        
        if not h_matrices:
            raise ValueError("At least one matrix must be provided")
        
        # Validate all matrices
        modes = h_matrices[0].shape[0]
        for i, h_matrix in enumerate(h_matrices):
            if h_matrix.ndim != 2 or h_matrix.shape[0] != h_matrix.shape[1]:
                raise ValueError(f"Matrix {i} must be square")
            if h_matrix.shape[0] != modes:
                raise ValueError(f"All matrices must have the same size")
            if not np.allclose(h_matrix, h_matrix.conj().T, atol=1e-12):
                raise ValueError(f"Matrix {i} must be Hermitian")
        
        super().__init__(modes)
        
        self.h_matrices = [h.copy() for h in h_matrices]
        self.order = len(h_matrices)
        self.expression = self._build_operator_expression()
    
    def _build_operator_expression(self) -> QuantumOperator:
        """
        Build the operator expression from the Hermitian matrices.
        
        For order n, the Jordan-Schwinger mapping is:
        Ô = Σ_{k_i,l_i} ∏_i (h_i)_{k_i,l_i} â†_{k_1} â†_{k_2} ... â†_{k_n} â_{l_n} ... â_{l_2} â_{l_1}
        
        The operator is built as a sum of all possible index combinations,
        with each term being a scaled product of creation and annihilation operators.
        
        Returns:
            QuantumOperator representing the mapped expression
        """
        terms = []
        
        # Generate all possible index combinations
        # For order n, we need n pairs of indices (k_i, l_i)
        index_ranges = [range(self.modes) for _ in range(2 * self.order)]
        
        for indices in itertools.product(*index_ranges):
            # Split indices into creation (k) and annihilation (l) parts
            k_indices = indices[:self.order]  # k_1, k_2, ..., k_n
            l_indices = indices[self.order:]  # l_1, l_2, ..., l_n
            
            # Compute the coefficient: ∏_i (h_i)_{k_i,l_i}
            coefficient = 1.0
            for i in range(self.order):
                coefficient *= self.h_matrices[i][k_indices[i], l_indices[i]]
            
            # Skip negligible coefficients
            if abs(coefficient) < 1e-15:
                continue
            
            # Build the operator product: â†_{k_1} â†_{k_2} ... â†_{k_n} â_{l_n} ... â_{l_2} â_{l_1}
            operators = []
            
            # Add creation operators (left to right: k_1, k_2, ..., k_n)
            for k in k_indices:
                operators.append(CreationOperator(k, self.modes))
            
            # Add annihilation operators (right to left: l_n, l_{n-1}, ..., l_1)
            for l in reversed(l_indices):
                operators.append(AnnihilationOperator(l, self.modes))
            
            # Create the operator product
            if len(operators) == 1:
                product = operators[0]
            else:
                product = OperatorProduct(operators)
            
            # Scale by coefficient
            if abs(coefficient - 1.0) > 1e-15:
                term = ScaledOperator(coefficient, product)
            else:
                term = product
            
            terms.append(term)
        
        if not terms:
            # All coefficients were negligible
            from .quantum_operators import ScalarOperator
            return ScalarOperator(0, self.modes)
        elif len(terms) == 1:
            return terms[0]
        else:
            return OperatorSum(terms)
    
    def evolve(self, state: Tuple[int, ...]) -> Dict[Tuple[int, ...], complex]:
        """
        Apply Jordan-Schwinger operator to a Fock state.
        
        This delegates to the built expression, which handles all the
        evolution logic through the OperatorSum/OperatorProduct framework.
        
        Args:
            state: Initial Fock state
            
        Returns:
            Dictionary with all possible final states and their amplitudes
        """
        return self.expression.evolve(state)
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """
        Convert to matrix representation.
        
        This delegates to the expression for consistency.
        """
        return self.expression.to_matrix(basis, state_to_index)
    
    def get_symbolic_form(self) -> str:
        """
        Get symbolic representation of the operator.
        
        For higher-order operators, this can be quite complex, so we provide
        a simplified representation showing the order and structure.
        
        Returns:
            String representation of the operator
        """
        if self.order == 1:
            # For first order, show detailed form
            return self._get_first_order_symbolic()
        else:
            # For higher order, show summary
            matrix_names = [f"h_{i+1}" for i in range(self.order)]
            return f"JS_order_{self.order}({' × '.join(matrix_names)})"
    
    def _get_first_order_symbolic(self) -> str:
        """Get detailed symbolic form for first-order operators."""
        h_matrix = self.h_matrices[0]
        terms = []
        
        for k in range(self.modes):
            for l in range(self.modes):
                coeff = h_matrix[k, l]
                
                if abs(coeff) < 1e-15:
                    continue
                
                # Format coefficient
                if abs(coeff.imag) < 1e-15:
                    coeff_str = f"{coeff.real:.3f}"
                else:
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
                    term_str = f"{coeff_str}n_{k}"
                else:
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
    
    def get_order(self) -> int:
        """Get the order of the Jordan-Schwinger mapping."""
        return self.order
    
    def get_matrices(self) -> List[np.ndarray]:
        """Get the original Hermitian matrices."""
        return [h.copy() for h in self.h_matrices]
    
    def __repr__(self) -> str:
        return f"JordanSchwinger_{self.order}({self.get_symbolic_form()})"


def jordan_schwinger_map(h_matrices: Union[np.ndarray, List[np.ndarray]], 
                        order: Optional[int] = None) -> JordanSchwingerOperator:
    """
    Apply Jordan-Schwinger mapping to Hermitian matrices of arbitrary order.
    
    The Jordan-Schwinger mapping generalizes to higher orders:
    
    Order 1: h → Ô = Σ_{k,l} h_{kl} â†_k â_l
    Order 2: h_1 × h_2 → Ô = Σ_{k,l,m,n} (h_1)_{kl} (h_2)_{mn} â†_k â†_m â_n â_l  
    Order n: h_1 × ... × h_n → Ô = Σ_{indices} ∏_i (h_i)_{k_i,l_i} ∏_j â†_{k_j} ∏_j â_{l_j}
    
    Properties preserved:
    - Hermiticity: if all h_i are Hermitian → Ô† = Ô
    - Linearity: in each matrix argument
    
    Args:
        h_matrices: Single Hermitian matrix or list of matrices
        order: Optional order specification (inferred from h_matrices if not given)
        
    Returns:
        JordanSchwingerOperator representing the mapped quantum operator
        
    Raises:
        ValueError: If matrices are not square, not Hermitian, or order mismatch
        
    Examples:
        >>> # First order (standard Jordan-Schwinger)
        >>> sigma_x = np.array([[0, 1], [1, 0]])
        >>> js_op1 = jordan_schwinger_map(sigma_x)
        >>> print(js_op1.get_symbolic_form())
        'a†_0·a_1 + a†_1·a_0'
        
        >>> # Second order
        >>> sigma_z = np.array([[1, 0], [0, -1]])
        >>> js_op2 = jordan_schwinger_map([sigma_x, sigma_z])  
        >>> print(js_op2.get_order())
        2
        
        >>> # Explicit order specification
        >>> js_op3 = jordan_schwinger_map([sigma_x, sigma_x, sigma_z], order=3)
    """
    # Handle single matrix input
    if isinstance(h_matrices, np.ndarray):
        h_matrices = [h_matrices]
    
    # Validate order if specified
    if order is not None:
        if order != len(h_matrices):
            raise ValueError(f"Specified order {order} doesn't match number of matrices {len(h_matrices)}")
        if order < 1:
            raise ValueError("Order must be at least 1")
    
    return JordanSchwingerOperator(h_matrices)


def pauli_operators(modes: int = 2) -> Dict[str, JordanSchwingerOperator]:
    """
    Generate Jordan-Schwinger mapped Pauli operators.
    
    Args:
        modes: Number of modes (should be 2 for standard Pauli matrices)
        
    Returns:
        Dictionary containing Pauli operators: {'X', 'Y', 'Z', 'I'}
        
    Raises:
        ValueError: If modes != 2
        
    Example:
        >>> pauli_ops = pauli_operators()
        >>> sigma_x = pauli_ops['X']
        >>> print(sigma_x.get_symbolic_form())
        'a†_0·a_1 + a†_1·a_0'
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

def generate_all_js_operators(basis: List[np.ndarray], order: int) -> List[JordanSchwingerOperator]:
    """
    Generate all possible Jordan-Schwinger operators of given order from a matrix basis.
    
    Args:
        basis: List of Hermitian matrices to use as basis
        order: Order of Jordan-Schwinger mapping (number of matrix factors)
        
    Returns:
        List of all possible JordanSchwingerOperator instances
        
    Example:
        >>> # Define a simple basis
        >>> sigma_x = np.array([[0, 1], [1, 0]])
        >>> sigma_z = np.array([[1, 0], [0, -1]]) 
        >>> basis = [sigma_x, sigma_z]
        >>> 
        >>> # Generate all 2nd order JS operators
        >>> operators = generate_all_js_operators(basis, order=2)
        >>> print(f"Generated {len(operators)} operators")
        Generated 4 operators
    """
    if order <= 0:
        raise ValueError("Order must be positive")
    
    if not basis:
        raise ValueError("Basis cannot be empty")
    
    operators = []
    
    # Generate all possible combinations of matrices from basis
    # For order k, we need k matrices (with repetition allowed)
    for matrix_indices in itertools.product(range(len(basis)), repeat=order):
        # Select the matrices for this combination
        matrices = [basis[i] for i in matrix_indices]
        
        # Create Jordan-Schwinger operator
        js_operator = jordan_schwinger_map(matrices)
        operators.append(js_operator)
    
    return operators


# Export main functions
__all__ = [
    'JordanSchwingerOperator',
    'jordan_schwinger_map',
    'pauli_operators',
    'generate_all_js_operators',
]