"""
Quantum Operator Framework - Main Classes

This module provides the core quantum operator classes for QOPTCRAFT.
All operators are basis-independent and can be converted to matrix
representations in any given Fock basis.

Classes:
    QuantumOperator: Abstract base class for all quantum operators
    CreationOperator: Photon creation operator â†_mode
    AnnihilationOperator: Photon annihilation operator â_mode  
    NumberOperator: Number operator n̂_mode = â†_mode â_mode
    ScalarOperator: Scalar multiple of identity operator
    ScaledOperator: Scalar multiple of another operator
    OperatorSum: Sum of multiple operators
    OperatorProduct: Product of multiple operators

Author: yangbc30 with claude sonnet 4
License: GPL-3.0
"""

import numpy as np
from typing import Tuple, List, Dict, Union, Optional
from abc import ABC, abstractmethod

# QOPTCRAFT imports
from qoptcraft.operators import creation_fock, annihilation_fock


class QuantumOperator(ABC):
    """
    Abstract base class for quantum operators.
    
    Operators are basis-independent mathematical objects that can be
    represented as matrices in any given Fock basis.
    
    Attributes:
        modes (int): Number of optical modes in the system
    """
    
    def __init__(self, modes: int):
        """
        Initialize a quantum operator.
        
        Args:
            modes: Number of optical modes
            
        Raises:
            ValueError: If modes is not a positive integer
        """
        if not isinstance(modes, int) or modes <= 0:
            raise ValueError(f"modes must be a positive integer, got {modes}")
        self.modes = modes
    
    @abstractmethod
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """
        Convert operator to matrix representation in given basis.
        
        Args:
            basis: List of Fock states defining the basis
            state_to_index: Optional mapping from states to indices for optimization
        
        Returns:
            Matrix representation of the operator
        """
        pass
    
    def eigenvalues(self, basis: List[Tuple[int, ...]]) -> np.ndarray:
        """
        Compute eigenvalues of the operator in given basis.
        
        Args:
            basis: Fock basis states
            
        Returns:
            Array of eigenvalues
        """
        matrix = self.to_matrix(basis)
        return np.linalg.eigvals(matrix)
    
    def is_hermitian(self, basis: List[Tuple[int, ...]], tol: float = 1e-12) -> bool:
        """
        Check if operator is Hermitian in given basis.
        
        Args:
            basis: Fock basis states
            tol: Numerical tolerance
            
        Returns:
            True if operator is Hermitian
        """
        matrix = self.to_matrix(basis)
        return np.allclose(matrix, matrix.conj().T, atol=tol)
    
    def expectation_value(self, state_vector: np.ndarray, 
                         basis: List[Tuple[int, ...]]) -> complex:
        """
        Compute expectation value ⟨ψ|Ô|ψ⟩.
        
        Args:
            state_vector: Normalized state vector in given basis
            basis: Fock basis states
            
        Returns:
            Expectation value
        """
        matrix = self.to_matrix(basis)
        return np.conj(state_vector) @ matrix @ state_vector
    
    def commutator(self, other: 'QuantumOperator') -> 'OperatorSum':
        """
        Compute commutator [A, B] = AB - BA.
        
        Args:
            other: Another quantum operator
            
        Returns:
            Commutator as an OperatorSum
        """
        if not isinstance(other, QuantumOperator):
            raise TypeError(f"Cannot compute commutator with {type(other)}")
        if other.modes != self.modes:
            raise ValueError(f"Mode mismatch: {self.modes} vs {other.modes}")
        
        return (self @ other) - (other @ self)
    
    def anticommutator(self, other: 'QuantumOperator') -> 'OperatorSum':
        """
        Compute anticommutator {A, B} = AB + BA.
        
        Args:
            other: Another quantum operator
            
        Returns:
            Anticommutator as an OperatorSum
        """
        if not isinstance(other, QuantumOperator):
            raise TypeError(f"Cannot compute anticommutator with {type(other)}")
        if other.modes != self.modes:
            raise ValueError(f"Mode mismatch: {self.modes} vs {other.modes}")
        
        return (self @ other) + (other @ self)
    
    # Operator arithmetic
    def __add__(self, other) -> 'OperatorSum':
        """Operator addition."""
        if isinstance(other, (int, float, complex)):
            return OperatorSum([self, ScalarOperator(other, self.modes)])
        if not isinstance(other, QuantumOperator):
            raise TypeError(f"Cannot add QuantumOperator and {type(other)}")
        if other.modes != self.modes:
            raise ValueError(f"Mode mismatch: {self.modes} vs {other.modes}")
        return OperatorSum([self, other])
    
    def __radd__(self, other: Union[int, float, complex]) -> 'OperatorSum':
        """Right addition for scalars."""
        return OperatorSum([ScalarOperator(other, self.modes), self])
    
    def __sub__(self, other) -> 'OperatorSum':
        """Operator subtraction."""
        return self + (-1.0 * other)
    
    def __rsub__(self, other: Union[int, float, complex]) -> 'OperatorSum':
        """Right subtraction for scalars."""
        return ScalarOperator(other, self.modes) + (-1.0 * self)
    
    def __mul__(self, scalar: Union[int, float, complex]) -> 'ScaledOperator':
        """Scalar multiplication."""
        if not isinstance(scalar, (int, float, complex)):
            raise TypeError(f"Can only multiply by scalar, got {type(scalar)}")
        return ScaledOperator(scalar, self)
    
    def __rmul__(self, scalar: Union[int, float, complex]) -> 'ScaledOperator':
        """Right scalar multiplication."""
        return ScaledOperator(scalar, self)
    
    def __truediv__(self, scalar: Union[int, float, complex]) -> 'ScaledOperator':
        """Scalar division."""
        if not isinstance(scalar, (int, float, complex)):
            raise TypeError(f"Can only divide by scalar, got {type(scalar)}")
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return ScaledOperator(1.0/scalar, self)
    
    def __neg__(self) -> 'ScaledOperator':
        """Unary negation."""
        return ScaledOperator(-1.0, self)
    
    def __matmul__(self, other: 'QuantumOperator') -> 'OperatorProduct':
        """Operator multiplication using @ symbol."""
        if not isinstance(other, QuantumOperator):
            raise TypeError(f"Cannot multiply QuantumOperator and {type(other)}")
        if other.modes != self.modes:
            raise ValueError(f"Mode mismatch: {self.modes} vs {other.modes}")
        return OperatorProduct([self, other])
    
    def __pow__(self, n: int) -> 'OperatorProduct':
        """Operator power: A^n = A @ A @ ... @ A (n times)."""
        if not isinstance(n, int) or n < 0:
            raise ValueError("Power must be a non-negative integer")
        if n == 0:
            return ScalarOperator(1.0, self.modes)  # Identity
        if n == 1:
            return self
        return OperatorProduct([self] * n)


class CreationOperator(QuantumOperator):
    """
    Photon creation operator â†_mode.
    
    Acts on Fock state |n⟩ to give √(n+1)|n+1⟩.
    
    Attributes:
        mode (int): Mode index (0-based)
    """
    
    def __init__(self, mode: int, modes: int):
        """
        Initialize creation operator.
        
        Args:
            mode: Mode index (0-based)
            modes: Total number of modes
            
        Raises:
            ValueError: If mode index is out of range
        """
        super().__init__(modes)
        if not (0 <= mode < modes):
            raise ValueError(f"Mode index {mode} out of range [0, {modes-1}]")
        self.mode = mode
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """Convert to matrix representation using QOPTCRAFT's creation_fock."""
        if state_to_index is None:
            state_to_index = {state: i for i, state in enumerate(basis)}
        
        dim = len(basis)
        matrix = np.zeros((dim, dim), dtype=complex)
        
        for j, initial_state in enumerate(basis):
            try:
                final_state, coeff = creation_fock(self.mode, initial_state)
                if final_state in state_to_index:
                    i = state_to_index[final_state]
                    matrix[i, j] = coeff
            except (IndexError, ValueError):
                # Handle cases where creation leads to states outside the basis
                continue
        
        return matrix
    
    def __repr__(self) -> str:
        return f"a†_{self.mode}"


class AnnihilationOperator(QuantumOperator):
    """
    Photon annihilation operator â_mode.
    
    Acts on Fock state |n⟩ to give √n|n-1⟩.
    
    Attributes:
        mode (int): Mode index (0-based)
    """
    
    def __init__(self, mode: int, modes: int):
        """
        Initialize annihilation operator.
        
        Args:
            mode: Mode index (0-based)
            modes: Total number of modes
            
        Raises:
            ValueError: If mode index is out of range
        """
        super().__init__(modes)
        if not (0 <= mode < modes):
            raise ValueError(f"Mode index {mode} out of range [0, {modes-1}]")
        self.mode = mode
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """Convert to matrix representation using QOPTCRAFT's annihilation_fock."""
        if state_to_index is None:
            state_to_index = {state: i for i, state in enumerate(basis)}
        
        dim = len(basis)
        matrix = np.zeros((dim, dim), dtype=complex)
        
        for j, initial_state in enumerate(basis):
            if initial_state[self.mode] > 0:  # Can only annihilate if photons exist
                try:
                    final_state, coeff = annihilation_fock(self.mode, initial_state)
                    if final_state in state_to_index:
                        i = state_to_index[final_state]
                        matrix[i, j] = coeff
                except (IndexError, ValueError):
                    continue
        
        return matrix
    
    def __repr__(self) -> str:
        return f"a_{self.mode}"


class NumberOperator(QuantumOperator):
    """
    Number operator n̂_mode = â†_mode â_mode.
    
    Counts photons in a given mode.
    
    Attributes:
        mode (int): Mode index (0-based)
    """
    
    def __init__(self, mode: int, modes: int):
        """
        Initialize number operator.
        
        Args:
            mode: Mode index (0-based)
            modes: Total number of modes
            
        Raises:
            ValueError: If mode index is out of range
        """
        super().__init__(modes)
        if not (0 <= mode < modes):
            raise ValueError(f"Mode index {mode} out of range [0, {modes-1}]")
        self.mode = mode
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """Convert to matrix representation (diagonal matrix)."""
        dim = len(basis)
        matrix = np.zeros((dim, dim), dtype=complex)
        
        for i, state in enumerate(basis):
            matrix[i, i] = state[self.mode]  # Diagonal elements are photon counts
        
        return matrix
    
    def __repr__(self) -> str:
        return f"n_{self.mode}"


class ScalarOperator(QuantumOperator):
    """
    Scalar multiple of identity operator.
    
    Represents c·Î where c is a complex scalar.
    
    Attributes:
        scalar (complex): Scalar coefficient
    """
    
    def __init__(self, scalar: Union[int, float, complex], modes: int):
        """
        Initialize scalar operator.
        
        Args:
            scalar: Scalar coefficient
            modes: Number of modes
        """
        super().__init__(modes)
        self.scalar = complex(scalar)
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """Convert to matrix representation (scalar times identity)."""
        dim = len(basis)
        return self.scalar * np.eye(dim, dtype=complex)
    
    def __repr__(self) -> str:
        if self.scalar == 1:
            return "I"
        elif self.scalar == -1:
            return "-I"
        elif self.scalar == 0:
            return "0"
        else:
            return f"{self.scalar}·I"


class ScaledOperator(QuantumOperator):
    """
    Scalar multiple of another operator.
    
    Represents c·Ô where c is a scalar and Ô is an operator.
    
    Attributes:
        scalar (complex): Scalar coefficient
        operator (QuantumOperator): Base operator
    """
    
    def __init__(self, scalar: Union[int, float, complex], operator: QuantumOperator):
        """
        Initialize scaled operator.
        
        Args:
            scalar: Scalar coefficient
            operator: Base operator
        """
        super().__init__(operator.modes)
        self.scalar = complex(scalar)
        self.operator = operator
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """Convert to matrix representation."""
        return self.scalar * self.operator.to_matrix(basis, state_to_index)
    
    def __repr__(self) -> str:
        if self.scalar == 1:
            return str(self.operator)
        elif self.scalar == -1:
            return f"-({self.operator})"
        elif self.scalar == 0:
            return "0"
        else:
            return f"{self.scalar}·({self.operator})"


class OperatorSum(QuantumOperator):
    """
    Sum of quantum operators.
    
    Represents Ô₁ + Ô₂ + ... + Ôₙ.
    
    Attributes:
        operators (List[QuantumOperator]): List of operators to sum
    """
    
    def __init__(self, operators: List[QuantumOperator]):
        """
        Initialize operator sum.
        
        Args:
            operators: List of operators to sum
            
        Raises:
            ValueError: If operator list is empty or modes don't match
        """
        if not operators:
            raise ValueError("Operator list cannot be empty")
        
        modes = operators[0].modes
        if not all(op.modes == modes for op in operators):
            raise ValueError("All operators must have the same number of modes")
        
        super().__init__(modes)
        # Flatten nested sums for efficiency
        self.operators = []
        for op in operators:
            if isinstance(op, OperatorSum):
                self.operators.extend(op.operators)
            else:
                self.operators.append(op)
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """Convert to matrix representation."""
        if state_to_index is None:
            state_to_index = {state: i for i, state in enumerate(basis)}
        
        result = np.zeros((len(basis), len(basis)), dtype=complex)
        for op in self.operators:
            result += op.to_matrix(basis, state_to_index)
        return result
    
    def __repr__(self) -> str:
        if not self.operators:
            return "0"
        
        result = str(self.operators[0])
        for op in self.operators[1:]:
            op_str = str(op)
            if op_str.startswith('-'):
                result += f" - {op_str[1:]}"
            else:
                result += f" + {op_str}"
        return result


class OperatorProduct(QuantumOperator):
    """
    Product of quantum operators.
    
    Represents Ô₁·Ô₂·...·Ôₙ (matrix multiplication order).
    
    Attributes:
        operators (List[QuantumOperator]): List of operators to multiply (left to right)
    """
    
    def __init__(self, operators: List[QuantumOperator]):
        """
        Initialize operator product.
        
        Args:
            operators: List of operators to multiply (left to right)
            
        Raises:
            ValueError: If operator list is empty or modes don't match
        """
        if not operators:
            raise ValueError("Operator list cannot be empty")
        
        modes = operators[0].modes
        if not all(op.modes == modes for op in operators):
            raise ValueError("All operators must have the same number of modes")
        
        super().__init__(modes)
        # Flatten nested products for efficiency
        self.operators = []
        for op in operators:
            if isinstance(op, OperatorProduct):
                self.operators.extend(op.operators)
            else:
                self.operators.append(op)
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """Convert to matrix representation."""
        if state_to_index is None:
            state_to_index = {state: i for i, state in enumerate(basis)}
        
        # Matrix multiplication from left to right
        result = self.operators[0].to_matrix(basis, state_to_index)
        for op in self.operators[1:]:
            result = result @ op.to_matrix(basis, state_to_index)
        return result
    
    def __repr__(self) -> str:
        if not self.operators:
            return "I"
        return "·".join(str(op) for op in self.operators)