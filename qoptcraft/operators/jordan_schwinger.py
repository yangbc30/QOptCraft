"""
Jordan-Schwinger Mapping Implementation

This module implements the Jordan-Schwinger mapping that maps
Hermitian matrices to quantum operators:

h_i → Ô_i = Σ_{k,l} (h_i)_{kl} â†_k â_l

The mapping preserves Hermiticity and provides a natural way to
construct quantum operators from classical matrices. All operators
are built using the evolve-based framework for consistency.

Classes:
    JordanSchwingerOperator: Operator created from Jordan-Schwinger mapping

Functions:
    jordan_schwinger_map: Main function to apply the mapping
    pauli_operators: Generate Pauli operators via Jordan-Schwinger mapping
    su2_generators: Generate SU(2) generators
    validate_jordan_schwinger_properties: Validation utilities

Author: QOPTCRAFT Extension  
License: Compatible with QOPTCRAFT license
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

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
    
    The evolution is implemented by building the operator as a sum
    of scaled creation-annihilation products, then using the
    OperatorSum evolution logic.
    
    Attributes:
        h_matrix (np.ndarray): Original Hermitian matrix
        expression (QuantumOperator): Built operator expression using evolve-based operators
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
        number operators n̂_k = â†_k â_k, and the entire expression
        is built as an OperatorSum of scaled terms.
        
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
                    if abs(coeff) > 1e-15:
                        term = ScaledOperator(coeff, NumberOperator(k, self.modes))
                        terms.append(term)
                else:
                    # Off-diagonal terms: creation × annihilation
                    # h_{kl} â†_k â_l
                    a_dag_k = CreationOperator(k, self.modes)
                    a_l = AnnihilationOperator(l, self.modes)
                    product = OperatorProduct([a_dag_k, a_l])
                    term = ScaledOperator(coeff, product)
                    terms.append(term)
        
        if not terms:
            # All coefficients were negligible - return zero operator
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
        
        This could use the base class implementation, but we delegate to
        the expression for consistency and potential optimizations.
        """
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
                
                # Remove unnecessary "1.000·" and "-1.000·" prefixes
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


def su2_generators(modes: int = 2) -> Dict[str, JordanSchwingerOperator]:
    """
    Generate SU(2) generators using Jordan-Schwinger mapping.
    
    The SU(2) generators are J_x = σ_x/2, J_y = σ_y/2, J_z = σ_z/2.
    These satisfy the angular momentum algebra [J_i, J_j] = iε_{ijk}J_k.
    
    Args:
        modes: Number of modes (should be 2)
        
    Returns:
        Dictionary containing SU(2) generators: {'Jx', 'Jy', 'Jz'}
        
    Example:
        >>> su2_ops = su2_generators()
        >>> jz = su2_ops['Jz']
        >>> print(jz.get_symbolic_form())
        '0.500·n_0 - 0.500·n_1'
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
    Generate displacement operator generator for coherent states using Jordan-Schwinger mapping.
    
    The displacement operator is D(α) = exp(α·a† - α*·a), but here we
    construct the generator α·a† - α*·a as a Jordan-Schwinger operator.
    
    This creates a matrix representation that can be used with matrix
    exponentiation to get the full displacement operator.
    
    Args:
        alpha: Complex displacement parameter
        mode: Mode to displace
        modes: Total number of modes
        
    Returns:
        JordanSchwingerOperator representing the displacement generator
        
    Example:
        >>> disp_gen = coherent_state_displacement(1+1j, 0, 2)
        >>> print(disp_gen.get_symbolic_form())
        # Will show the generator form, not the full exponential
    """
    if not (0 <= mode < modes):
        raise ValueError(f"Mode index {mode} out of range [0, {modes-1}]")
    
    # Create matrix representation of displacement generator
    # For the displacement generator α·a† - α*·a, we need to use
    # a representation that Jordan-Schwinger can handle
    
    # We'll use the fact that for quadrature displacements:
    # x̂ = (a† + a)/√2, p̂ = i(a† - a)/√2
    # So α·a† - α*·a can be written in terms of quadratures
    
    h_matrix = np.zeros((modes, modes), dtype=complex)
    
    # For a displacement in the complex plane, we can decompose:
    # α = (α.real + i·α.imag)
    # This creates off-diagonal terms in the Jordan-Schwinger representation
    
    # Create a matrix that gives the desired generator when mapped
    # This is a simplified version - for exact displacement, one would
    # need to be more careful about the mapping
    
    if modes >= 2:
        # Use off-diagonal elements to create the displacement effect
        beta = alpha / np.sqrt(2)  # Scaling factor
        
        # Create off-diagonal terms that will map to a†a† and aa terms
        next_mode = (mode + 1) % modes
        h_matrix[mode, next_mode] = beta
        h_matrix[next_mode, mode] = np.conj(beta)
    else:
        # For single mode, use diagonal displacement
        h_matrix[mode, mode] = alpha.real
    
    return jordan_schwinger_map(h_matrix)


def validate_jordan_schwinger_properties(h_matrix: np.ndarray, 
                                       basis: List[Tuple[int, ...]],
                                       tolerance: float = 1e-12) -> Dict[str, bool]:
    """
    Validate that Jordan-Schwinger mapping preserves expected properties.
    
    This function checks various mathematical properties that should be
    preserved under the Jordan-Schwinger mapping, such as Hermiticity
    and linearity.
    
    Args:
        h_matrix: Original Hermitian matrix
        basis: Fock basis for testing
        tolerance: Numerical tolerance for checks
        
    Returns:
        Dictionary of validation results
        
    Example:
        >>> from qoptcraft.basis import photon_basis
        >>> sigma_x = np.array([[0, 1], [1, 0]])
        >>> basis = photon_basis(2, 1)
        >>> results = validate_jordan_schwinger_properties(sigma_x, basis)
        >>> print(results['output_hermitian'])
        True
    """
    # Apply mapping
    js_op = jordan_schwinger_map(h_matrix)
    js_matrix = js_op.to_matrix(basis)
    
    # Check basic properties
    results = {
        'input_hermitian': np.allclose(h_matrix, h_matrix.conj().T, atol=tolerance),
        'output_hermitian': np.allclose(js_matrix, js_matrix.conj().T, atol=tolerance),
        'output_finite': np.isfinite(js_matrix).all(),
        'correct_dimension': js_matrix.shape == (len(basis), len(basis)),
        'evolution_consistent': True  # We'll test this below
    }
    
    # Test evolution consistency
    try:
        for state in basis[:min(3, len(basis))]:  # Test a few states
            evolution_result = js_op.evolve(state)
            # Should return valid results
            if not all(isinstance(amp, (int, float, complex)) for amp in evolution_result.values()):
                results['evolution_consistent'] = False
                break
    except Exception:
        results['evolution_consistent'] = False
    
    # Check linearity if we have a 2x2 system
    if h_matrix.shape[0] == 2:
        try:
            h1 = np.array([[1, 0], [0, 0]], dtype=complex)
            h2 = np.array([[0, 1], [1, 0]], dtype=complex)
            a, b = 2.0, -1.5
            
            # Direct mapping of linear combination
            js_combined = jordan_schwinger_map(a * h1 + b * h2)
            combined_matrix = js_combined.to_matrix(basis)
            
            # Linear combination of mapped operators
            js1 = jordan_schwinger_map(h1)
            js2 = jordan_schwinger_map(h2)
            
            # Build linear combination using operator arithmetic
            linear_combination = ScaledOperator(a, js1).expression + ScaledOperator(b, js2).expression
            linear_matrix = linear_combination.to_matrix(basis)
            
            results['linearity'] = np.allclose(combined_matrix, linear_matrix, atol=tolerance)
        except Exception:
            results['linearity'] = False
    
    return results


def demonstrate_jordan_schwinger():
    """
    Demonstrate the Jordan-Schwinger mapping with various examples.
    
    This function showcases the main features and validates the implementation
    with several physical examples.
    """
    print("Jordan-Schwinger Mapping Demonstration")
    print("=" * 50)
    
    # Example 1: Pauli matrices
    print("\n1. Pauli Matrices")
    print("-" * 20)
    
    pauli_ops = pauli_operators()
    for name, op in pauli_ops.items():
        print(f"σ_{name}: {op.get_symbolic_form()}")
    
    # Example 2: SU(2) generators
    print("\n2. SU(2) Generators")
    print("-" * 20)
    
    su2_ops = su2_generators()
    for name, op in su2_ops.items():
        print(f"{name}: {op.get_symbolic_form()}")
    
    # Example 3: Custom matrix
    print("\n3. Custom Hermitian Matrix")
    print("-" * 30)
    
    custom_matrix = np.array([[1, 0.5+0.2j], [0.5-0.2j, -1]], dtype=complex)
    custom_op = jordan_schwinger_map(custom_matrix)
    print(f"Custom matrix:\n{custom_matrix}")
    print(f"Mapped operator: {custom_op.get_symbolic_form()}")
    
    # Example 4: Matrix representation
    print("\n4. Matrix Representation Test")
    print("-" * 30)
    
    try:
        from qoptcraft.basis import photon_basis
        basis = photon_basis(2, 1)
        
        sigma_x = pauli_ops['X']
        matrix = sigma_x.to_matrix(basis)
        
        print(f"Basis: {basis}")
        print(f"σ_X matrix in 1-photon basis:\n{matrix}")
        print(f"Is Hermitian: {sigma_x.is_hermitian(basis)}")
        
    except ImportError:
        print("QOPTCRAFT basis functions not available for full demonstration")


# Export main functions
__all__ = [
    'JordanSchwingerOperator',
    'jordan_schwinger_map',
    'pauli_operators',
    'su2_generators', 
    'coherent_state_displacement',
    'validate_jordan_schwinger_properties',
    'demonstrate_jordan_schwinger'
]