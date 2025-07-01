"""
Operator Arithmetic and Convenience Functions

This module provides convenience functions for creating quantum operators
and performing common operator arithmetic operations. It serves as the
main user interface for the operator framework.

Functions:
    creation_op, a_dag: Create creation operators
    annihilation_op, a: Create annihilation operators  
    number_op, n: Create number operators
    total_photon_number: Create total photon number operator
    position_op, momentum_op: Create quadrature operators
    coherent_displacement: Create coherent displacement operators
    squeeze_operator: Create squeezing operators

Author: yangbc30 with claude sonnet 4
License: GPL-3.0
"""

import numpy as np
from typing import List, Union, Optional
import cmath

from .quantum_operators import (
    QuantumOperator, CreationOperator, AnnihilationOperator, 
    NumberOperator, OperatorSum, ScaledOperator, OperatorProduct,
    ScalarOperator
)


# Basic operator creation functions
def creation_op(mode: int, modes: int) -> CreationOperator:
    """
    Create a creation operator â†_mode.
    
    Args:
        mode: Mode index (0-based)
        modes: Total number of modes
        
    Returns:
        Creation operator for the specified mode
        
    Example:
        >>> a0_dag = creation_op(0, 2)  # â†_0 for 2-mode system
        >>> print(a0_dag)
        a†_0
    """
    return CreationOperator(mode, modes)


def annihilation_op(mode: int, modes: int) -> AnnihilationOperator:
    """
    Create an annihilation operator â_mode.
    
    Args:
        mode: Mode index (0-based)
        modes: Total number of modes
        
    Returns:
        Annihilation operator for the specified mode
        
    Example:
        >>> a1 = annihilation_op(1, 2)  # â_1 for 2-mode system
        >>> print(a1)
        a_1
    """
    return AnnihilationOperator(mode, modes)


def number_op(mode: int, modes: int) -> NumberOperator:
    """
    Create a number operator n̂_mode = â†_mode â_mode.
    
    Args:
        mode: Mode index (0-based)
        modes: Total number of modes
        
    Returns:
        Number operator for the specified mode
        
    Example:
        >>> n0 = number_op(0, 2)  # n̂_0 for 2-mode system
        >>> print(n0)
        n_0
    """
    return NumberOperator(mode, modes)


# Short aliases for convenience
a_dag = creation_op
a = annihilation_op
n = number_op


# Composite operators
def total_photon_number(modes: int) -> OperatorSum:
    """
    Create total photon number operator N̂ = Σᵢ n̂ᵢ.
    
    Args:
        modes: Number of modes
        
    Returns:
        Total photon number operator
        
    Example:
        >>> N = total_photon_number(3)  # N̂ = n̂_0 + n̂_1 + n̂_2
        >>> print(N)
        n_0 + n_1 + n_2
    """
    if modes <= 0:
        raise ValueError("Number of modes must be positive")
    
    number_operators = [NumberOperator(i, modes) for i in range(modes)]
    return OperatorSum(number_operators)


def position_op(mode: int, modes: int) -> OperatorSum:
    """
    Create position quadrature operator x̂ = (â† + â)/√2.
    
    Args:
        mode: Mode index
        modes: Total number of modes
        
    Returns:
        Position quadrature operator
        
    Example:
        >>> x0 = position_op(0, 2)  # x̂_0 = (â†_0 + â_0)/√2
    """
    a_dag_mode = CreationOperator(mode, modes)
    a_mode = AnnihilationOperator(mode, modes)
    sqrt2_inv = 1.0 / np.sqrt(2)
    
    return sqrt2_inv * (a_dag_mode + a_mode)


def momentum_op(mode: int, modes: int) -> OperatorSum:
    """
    Create momentum quadrature operator p̂ = i(â† - â)/√2.
    
    Args:
        mode: Mode index
        modes: Total number of modes
        
    Returns:
        Momentum quadrature operator
        
    Example:
        >>> p0 = momentum_op(0, 2)  # p̂_0 = i(â†_0 - â_0)/√2
    """
    a_dag_mode = CreationOperator(mode, modes)
    a_mode = AnnihilationOperator(mode, modes)
    coeff = 1j / np.sqrt(2)
    
    return coeff * (a_dag_mode - a_mode)


def coherent_displacement(alpha: complex, mode: int, modes: int) -> OperatorSum:
    """
    Create coherent displacement generator α·â† - α*·â.
    
    The displacement operator is D(α) = exp(α·â† - α*·â).
    This function returns the generator (exponent).
    
    Args:
        alpha: Complex displacement amplitude
        mode: Mode index
        modes: Total number of modes
        
    Returns:
        Displacement generator
        
    Example:
        >>> disp_gen = coherent_displacement(1+1j, 0, 2)
        >>> # For actual displacement operator, compute matrix exponential
    """
    a_dag_mode = CreationOperator(mode, modes)
    a_mode = AnnihilationOperator(mode, modes)
    
    return alpha * a_dag_mode - np.conj(alpha) * a_mode


def squeeze_operator(xi: complex, mode1: int, mode2: int, modes: int) -> OperatorSum:
    """
    Create two-mode squeezing generator ξ·â₁â₂ - ξ*·â†₁â†₂.
    
    The squeezing operator is S(ξ) = exp(ξ·â₁â₂ - ξ*·â†₁â†₂).
    This function returns the generator (exponent).
    
    Args:
        xi: Complex squeezing parameter
        mode1: First mode index
        mode2: Second mode index  
        modes: Total number of modes
        
    Returns:
        Squeezing generator
        
    Example:
        >>> squeeze_gen = squeeze_operator(0.1, 0, 1, 2)
        >>> # Generates two-mode squeezed states
    """
    if mode1 == mode2:
        raise ValueError("Two-mode squeezing requires different modes")
    
    a1 = AnnihilationOperator(mode1, modes)
    a2 = AnnihilationOperator(mode2, modes)
    a1_dag = CreationOperator(mode1, modes)
    a2_dag = CreationOperator(mode2, modes)
    
    return xi * (a1 @ a2) - np.conj(xi) * (a1_dag @ a2_dag)


def beamsplitter_operator(theta: float, phi: float, mode1: int, mode2: int, modes: int) -> OperatorSum:
    """
    Create beamsplitter interaction generator.
    
    The beamsplitter operator is BS(θ,φ) = exp(θ·e^(iφ)·â†₁â₂ - θ·e^(-iφ)·â†₂â₁).
    This function returns the generator.
    
    Args:
        theta: Coupling strength
        phi: Phase angle
        mode1: First mode index
        mode2: Second mode index
        modes: Total number of modes
        
    Returns:
        Beamsplitter generator
        
    Example:
        >>> bs_gen = beamsplitter_operator(np.pi/4, 0, 0, 1, 2)
        >>> # 50:50 beamsplitter between modes 0 and 1
    """
    if mode1 == mode2:
        raise ValueError("Beamsplitter requires different modes")
    
    a1_dag = CreationOperator(mode1, modes)
    a2_dag = CreationOperator(mode2, modes)
    a1 = AnnihilationOperator(mode1, modes)
    a2 = AnnihilationOperator(mode2, modes)
    
    # Coupling coefficients
    coeff_pos = theta * cmath.exp(1j * phi)
    coeff_neg = theta * cmath.exp(-1j * phi)
    
    return coeff_pos * (a1_dag @ a2) - coeff_neg * (a2_dag @ a1)


def kerr_operator(chi: float, mode: int, modes: int) -> OperatorProduct:
    """
    Create Kerr nonlinearity operator χ·n̂(n̂-1).
    
    Args:
        chi: Kerr coefficient
        mode: Mode index
        modes: Total number of modes
        
    Returns:
        Kerr interaction operator
        
    Example:
        >>> kerr = kerr_operator(0.01, 0, 2)  # Self-phase modulation in mode 0
    """
    n_mode = NumberOperator(mode, modes)
    identity = ScalarOperator(1, modes)
    
    # n̂(n̂-1) = n̂² - n̂
    n_squared = n_mode @ n_mode
    kerr_term = chi * (n_squared - n_mode)
    
    return kerr_term


def cross_kerr_operator(chi: float, mode1: int, mode2: int, modes: int) -> OperatorProduct:
    """
    Create cross-Kerr interaction operator χ·n̂₁n̂₂.
    
    Args:
        chi: Cross-Kerr coefficient
        mode1: First mode index
        mode2: Second mode index
        modes: Total number of modes
        
    Returns:
        Cross-Kerr interaction operator
        
    Example:
        >>> cross_kerr = cross_kerr_operator(0.001, 0, 1, 2)
        >>> # Cross-phase modulation between modes 0 and 1
    """
    if mode1 == mode2:
        raise ValueError("Cross-Kerr requires different modes")
    
    n1 = NumberOperator(mode1, modes)
    n2 = NumberOperator(mode2, modes)
    
    return chi * (n1 @ n2)


def jaynes_cummings_operator(g: float, omega_atom: float, mode: int, modes: int) -> OperatorSum:
    """
    Create Jaynes-Cummings interaction operator.
    
    H_JC = ω_atom·σ_z/2 + g(â†σ₋ + âσ₊)
    
    This is a simplified version that represents the interaction term only.
    The atomic operators σ₊, σ₋ are represented using Jordan-Schwinger mapping
    in a two-level system.
    
    Args:
        g: Coupling strength
        omega_atom: Atomic transition frequency
        mode: Cavity mode index
        modes: Total number of modes (must include atom as one mode)
        
    Returns:
        Jaynes-Cummings Hamiltonian
        
    Note:
        This assumes the atom is represented as mode (modes-1)
    """
    if modes < 2:
        raise ValueError("Jaynes-Cummings requires at least 2 modes (cavity + atom)")
    
    # Cavity operators
    a_cavity = AnnihilationOperator(mode, modes)
    a_dag_cavity = CreationOperator(mode, modes)
    
    # Atom operators (assume atom is the last mode)
    atom_mode = modes - 1
    sigma_plus = CreationOperator(atom_mode, modes)   # σ₊ ≈ â†_atom
    sigma_minus = AnnihilationOperator(atom_mode, modes)  # σ₋ ≈ â_atom
    sigma_z = 2 * NumberOperator(atom_mode, modes) - ScalarOperator(1, modes)  # σ_z ≈ 2n̂ - 1
    
    # Jaynes-Cummings Hamiltonian
    atom_energy = (omega_atom / 2) * sigma_z
    interaction = g * (a_dag_cavity @ sigma_minus + a_cavity @ sigma_plus)
    
    return atom_energy + interaction


def parametric_amplifier_operator(epsilon: complex, mode1: int, mode2: int, modes: int) -> OperatorSum:
    """
    Create parametric amplification operator ε·â†₁â†₂ + ε*·â₁â₂.
    
    Args:
        epsilon: Complex pump strength
        mode1: First mode (signal)
        mode2: Second mode (idler)  
        modes: Total number of modes
        
    Returns:
        Parametric amplification operator
        
    Example:
        >>> para_amp = parametric_amplifier_operator(0.1j, 0, 1, 2)
        >>> # Parametric down-conversion
    """
    if mode1 == mode2:
        raise ValueError("Parametric amplification requires different modes")
    
    a1_dag = CreationOperator(mode1, modes)
    a2_dag = CreationOperator(mode2, modes)
    a1 = AnnihilationOperator(mode1, modes)
    a2 = AnnihilationOperator(mode2, modes)
    
    return epsilon * (a1_dag @ a2_dag) + np.conj(epsilon) * (a1 @ a2)


def hopping_operator(t: Union[float, complex], mode1: int, mode2: int, modes: int) -> OperatorSum:
    """
    Create photon hopping operator t·(â†₁â₂ + â†₂â₁).
    
    Args:
        t: Hopping amplitude (tunneling strength)
        mode1: First mode
        mode2: Second mode
        modes: Total number of modes
        
    Returns:
        Hopping operator
        
    Example:
        >>> hop = hopping_operator(0.1, 0, 1, 3)  # Hopping between modes 0 and 1
    """
    if mode1 == mode2:
        raise ValueError("Hopping requires different modes")
    
    a1_dag = CreationOperator(mode1, modes)
    a2_dag = CreationOperator(mode2, modes)
    a1 = AnnihilationOperator(mode1, modes)
    a2 = AnnihilationOperator(mode2, modes)
    
    return t * (a1_dag @ a2 + a2_dag @ a1)


def ring_lattice_operator(t: Union[float, complex], modes: int) -> OperatorSum:
    """
    Create ring lattice hopping operator with periodic boundary conditions.
    
    H_ring = t·Σᵢ(â†ᵢâᵢ₊₁ + â†ᵢ₊₁âᵢ)
    
    Args:
        t: Hopping amplitude
        modes: Number of modes (sites in the ring)
        
    Returns:
        Ring lattice Hamiltonian
        
    Example:
        >>> ring = ring_lattice_operator(0.1, 4)  # 4-site ring
    """
    if modes < 3:
        raise ValueError("Ring lattice requires at least 3 modes")
    
    hopping_terms = []
    for i in range(modes):
        j = (i + 1) % modes  # Periodic boundary conditions
        hopping_terms.append(hopping_operator(t, i, j, modes))
    
    return OperatorSum(hopping_terms)


def linear_chain_operator(t: Union[float, complex], modes: int) -> OperatorSum:
    """
    Create linear chain hopping operator with open boundary conditions.
    
    H_chain = t·Σᵢ₌₀ᴺ⁻²(â†ᵢâᵢ₊₁ + â†ᵢ₊₁âᵢ)
    
    Args:
        t: Hopping amplitude
        modes: Number of modes (sites in the chain)
        
    Returns:
        Linear chain Hamiltonian
        
    Example:
        >>> chain = linear_chain_operator(0.1, 5)  # 5-site chain
    """
    if modes < 2:
        raise ValueError("Linear chain requires at least 2 modes")
    
    hopping_terms = []
    for i in range(modes - 1):
        hopping_terms.append(hopping_operator(t, i, i + 1, modes))
    
    return OperatorSum(hopping_terms)


# Utility functions
def commutator(op1: QuantumOperator, op2: QuantumOperator) -> OperatorSum:
    """
    Compute commutator [A, B] = AB - BA.
    
    Args:
        op1: First operator
        op2: Second operator
        
    Returns:
        Commutator [op1, op2]
    """
    return op1.commutator(op2)


def anticommutator(op1: QuantumOperator, op2: QuantumOperator) -> OperatorSum:
    """
    Compute anticommutator {A, B} = AB + BA.
    
    Args:
        op1: First operator
        op2: Second operator
        
    Returns:
        Anticommutator {op1, op2}
    """
    return op1.anticommutator(op2)


def dagger(op: QuantumOperator, basis: List[tuple]) -> QuantumOperator:
    """
    Compute Hermitian conjugate of an operator.
    
    Note: This is a placeholder implementation. For a complete implementation,
    we would need to define how each operator type transforms under dagger.
    
    Args:
        op: Operator to take Hermitian conjugate of
        basis: Basis for matrix representation
        
    Returns:
        Hermitian conjugate operator (approximation via matrix)
    """
    # This is a simplified implementation via matrix representation
    matrix = op.to_matrix(basis)
    dagger_matrix = matrix.conj().T
    
    # Convert back to operator (this is not exact for symbolic operations)
    # In a full implementation, we would have proper dagger rules for each operator type
    from .jordan_schwinger import jordan_schwinger_map
    
    # This is an approximation - proper implementation would need symbolic rules
    return jordan_schwinger_map(dagger_matrix)


# Export all functions
__all__ = [
    # Basic operators
    'creation_op', 'annihilation_op', 'number_op',
    'a_dag', 'a', 'n',
    
    # Composite operators
    'total_photon_number', 'position_op', 'momentum_op',
    'coherent_displacement', 'squeeze_operator', 'beamsplitter_operator',
    'kerr_operator', 'cross_kerr_operator', 'jaynes_cummings_operator',
    'parametric_amplifier_operator', 'hopping_operator',
    'ring_lattice_operator', 'linear_chain_operator',
    
    # Utility functions
    'commutator', 'anticommutator', 'dagger'
]