"""
Quantum Operator Framework - Complete Implementation

This module provides the core quantum operator classes for QOPTCRAFT.
All operators are basis-independent and can be converted to matrix
representations in any given Fock basis through state evolution.

The key design principle is that every operator implements an evolve() method
that describes how it transforms quantum states, and the to_matrix() method
is built on top of this evolution logic.

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
from collections import defaultdict

# QOPTCRAFT imports
from qoptcraft.operators import creation_fock, annihilation_fock


class QuantumOperator(ABC):
    """
    Abstract base class for quantum operators.
    
    Operators are basis-independent mathematical objects that can be
    represented as matrices in any given Fock basis through state evolution.
    
    The key method is evolve(state) which returns a dictionary of
    {final_state: amplitude} pairs representing the result of applying
    the operator to the given state.
    
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
    def evolve(self, state: Tuple[int, ...]) -> Dict[Tuple[int, ...], complex]:
        """
        Apply the operator to a Fock state and return all possible outcomes.
        
        This is the core method that defines how the operator acts on quantum states.
        It should return a dictionary mapping final states to their amplitudes.
        
        Args:
            state: Initial Fock state as tuple (n_0, n_1, ..., n_{modes-1})
            
        Returns:
            Dictionary mapping final states to their amplitudes:
            {final_state: amplitude, ...}
            
        Example:
            >>> a_dag = CreationOperator(0, 2)
            >>> result = a_dag.evolve((1, 0))  # Apply â†_0 to |1,0⟩
            >>> # Returns {(2, 0): sqrt(2)}
        """
        pass
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """
        Convert operator to matrix representation using state evolution.
        
        This is the universal implementation that works for all operator types
        by using the evolve method. For each initial state in the basis,
        we apply the operator and collect the amplitudes for all final states.
        
        Args:
            basis: List of Fock states defining the basis
            state_to_index: Optional mapping from states to indices for optimization
        
        Returns:
            Matrix representation of the operator
        """
        if state_to_index is None:
            state_to_index = {state: i for i, state in enumerate(basis)}
        
        dim = len(basis)
        matrix = np.zeros((dim, dim), dtype=complex)
        
        # For each initial state |j⟩ in the basis
        for j, initial_state in enumerate(basis):
            # Apply operator and get all possible final states
            final_states = self.evolve(initial_state)
            
            # Fill in matrix elements ⟨i|Ô|j⟩
            for final_state, amplitude in final_states.items():
                if abs(amplitude) > 1e-15 and final_state in state_to_index:
                    i = state_to_index[final_state]
                    matrix[i, j] = amplitude
        
        return matrix
    
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
    
    In fixed-photon-number basis, this operator typically gives
    a zero matrix since creation increases photon number.
    
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
    
    def evolve(self, state: Tuple[int, ...]) -> Dict[Tuple[int, ...], complex]:
        """
        Apply creation operator â†_mode to a Fock state.
        
        Args:
            state: Initial Fock state
            
        Returns:
            Dictionary with single entry {final_state: sqrt(n+1)}
        """
        try:
            final_state, coeff = creation_fock(self.mode, state)
            return {final_state: coeff}
        except (IndexError, ValueError):
            # Handle any errors gracefully
            return {}
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """
        Convert creation operator to matrix representation.
        
        This is optimized for the case where we know creation operators
        often give zero matrices in fixed-photon-number basis.
        """
        if state_to_index is None:
            state_to_index = {state: i for i, state in enumerate(basis)}
        
        dim = len(basis)
        matrix = np.zeros((dim, dim), dtype=complex)
        
        for j, initial_state in enumerate(basis):
            try:
                final_state, coeff = creation_fock(self.mode, initial_state)
                # Only add to matrix if final state is in the basis
                if final_state in state_to_index:
                    i = state_to_index[final_state]
                    matrix[i, j] = coeff
                # If final_state not in basis, matrix element remains 0
            except (IndexError, ValueError):
                continue
        
        return matrix
    
    def __repr__(self) -> str:
        return f"a†_{self.mode}"


class AnnihilationOperator(QuantumOperator):
    """
    Photon annihilation operator â_mode.
    
    Acts on Fock state |n⟩ to give √n|n-1⟩.
    
    In fixed-photon-number basis, this operator typically gives
    a zero matrix since annihilation decreases photon number.
    
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
    
    def evolve(self, state: Tuple[int, ...]) -> Dict[Tuple[int, ...], complex]:
        """
        Apply annihilation operator â_mode to a Fock state.
        
        Args:
            state: Initial Fock state
            
        Returns:
            Dictionary with single entry {final_state: sqrt(n)} or empty if n=0
        """
        if state[self.mode] == 0:
            # Cannot annihilate from empty mode - optimization
            return {}
        
        try:
            final_state, coeff = annihilation_fock(self.mode, state)
            return {final_state: coeff}
        except (IndexError, ValueError):
            return {}
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """
        Convert annihilation operator to matrix representation.
        
        This is optimized for the case where we know annihilation operators
        often give zero matrices in fixed-photon-number basis.
        """
        if state_to_index is None:
            state_to_index = {state: i for i, state in enumerate(basis)}
        
        dim = len(basis)
        matrix = np.zeros((dim, dim), dtype=complex)
        
        for j, initial_state in enumerate(basis):
            if initial_state[self.mode] > 0:  # Can only annihilate if photons exist
                try:
                    final_state, coeff = annihilation_fock(self.mode, initial_state)
                    # Only add to matrix if final state is in the basis
                    if final_state in state_to_index:
                        i = state_to_index[final_state]
                        matrix[i, j] = coeff
                    # If final_state not in basis, matrix element remains 0
                except (IndexError, ValueError):
                    continue
        
        return matrix
    
    def __repr__(self) -> str:
        return f"a_{self.mode}"


class NumberOperator(QuantumOperator):
    """
    Number operator n̂_mode = â†_mode â_mode.
    
    Counts photons in a given mode. This is a diagonal operator
    that works correctly in any basis since it doesn't change
    the total photon number.
    
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
    
    def evolve(self, state: Tuple[int, ...]) -> Dict[Tuple[int, ...], complex]:
        """
        Apply number operator n̂_mode to a Fock state.
        
        Args:
            state: Initial Fock state
            
        Returns:
            Dictionary with single entry {state: n_mode} (diagonal operator)
        """
        photon_count = state[self.mode]
        return {state: complex(photon_count)}
    
    def to_matrix(self, basis: List[Tuple[int, ...]], 
                  state_to_index: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """
        Convert number operator to matrix representation (diagonal matrix).
        
        This works correctly in any basis since the number operator
        doesn't change photon number - it's always diagonal.
        """
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
    This operator simply multiplies every state by the scalar.
    
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
    
    def evolve(self, state: Tuple[int, ...]) -> Dict[Tuple[int, ...], complex]:
        """
        Apply scalar operator c·Î to a Fock state.
        
        Args:
            state: Initial Fock state
            
        Returns:
            Dictionary with single entry {state: c} (identity operation)
        """
        return {state: self.scalar}
    
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
    
    def evolve(self, state: Tuple[int, ...]) -> Dict[Tuple[int, ...], complex]:
        """
        Apply scaled operator c·Ô to a Fock state.
        
        Args:
            state: Initial Fock state
            
        Returns:
            Dictionary with all final states scaled by c
        """
        base_result = self.operator.evolve(state)
        return {final_state: self.scalar * amplitude 
                for final_state, amplitude in base_result.items()}
    
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
    
    The evolution of a sum is the sum of individual evolutions,
    with amplitudes added for states that appear in multiple results.
    
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
    
    def evolve(self, state: Tuple[int, ...]) -> Dict[Tuple[int, ...], complex]:
        """
        Apply operator sum (Ô₁ + Ô₂ + ... + Ôₙ) to a Fock state.
        
        The result is the sum of individual operator evolution results.
        If multiple operators produce the same final state, their amplitudes
        are added together.
        
        Args:
            state: Initial Fock state
            
        Returns:
            Dictionary combining all final states from all operators
        """
        combined_result = defaultdict(complex)
        
        for operator in self.operators:
            op_result = operator.evolve(state)
            for final_state, amplitude in op_result.items():
                combined_result[final_state] += amplitude
        
        # Remove near-zero amplitudes to avoid numerical noise
        return {final_state: amplitude 
                for final_state, amplitude in combined_result.items()
                if abs(amplitude) > 1e-15}
    
    def __repr__(self) -> str:
        if not self.operators:
            return "0"
        
        # Build string representation with proper signs
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
    
    Represents Ô₁·Ô₂·...·Ôₙ applied sequentially (right to left in quantum mechanics).
    
    The evolution proceeds by applying operators one by one from right to left,
    with the output of one operator becoming the input to the next.
    
    Attributes:
        operators (List[QuantumOperator]): List of operators to multiply (left to right in notation)
    """
    
    def __init__(self, operators: List[QuantumOperator]):
        """
        Initialize operator product.
        
        Args:
            operators: List of operators to multiply (left to right in notation)
                      Note: In quantum mechanics, rightmost operator acts first
            
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
    
    def evolve(self, state: Tuple[int, ...]) -> Dict[Tuple[int, ...], complex]:
        """
        Apply operator product Ô₁·Ô₂·...·Ôₙ to a Fock state.
        
        Operators are applied sequentially from right to left (quantum mechanics convention).
        At each step, we maintain a dictionary of {state: amplitude} pairs representing
        the current superposition, and apply the next operator to each component.
        
        Args:
            state: Initial Fock state
            
        Returns:
            Dictionary with all possible final states and their amplitudes
        """
        # Start with the initial state and amplitude 1
        current_states = {state: complex(1.0)}
        
        # Apply operators from right to left (quantum mechanics convention)
        for operator in reversed(self.operators):
            new_states = defaultdict(complex)
            
            # Apply current operator to each state in the superposition
            for current_state, current_amplitude in current_states.items():
                if abs(current_amplitude) < 1e-15:
                    continue  # Skip negligible amplitudes for optimization
                
                # Apply operator to current state
                op_result = operator.evolve(current_state)
                
                # Add results to the new superposition
                for final_state, op_amplitude in op_result.items():
                    new_states[final_state] += current_amplitude * op_amplitude
            
            # Update current states for next iteration
            current_states = dict(new_states)
        
        # Remove near-zero amplitudes
        return {final_state: amplitude 
                for final_state, amplitude in current_states.items()
                if abs(amplitude) > 1e-15}
    
    def __repr__(self) -> str:
        if not self.operators:
            return "I"
        return "·".join(str(op) for op in self.operators)