"""
QOPTCRAFT Operators Module

Quantum operators framework with Jordan-Schwinger mapping support.
"""

# Import existing QOPTCRAFT operators
from .ladder import creation_fock, annihilation_fock
from .adjoint import adjoint_evol
from .qft import qft, qft_inv
from .quasiunitary import random_quasiunitary

# Import quantum operator framework
from .quantum_operators import (
    QuantumOperator,
    CreationOperator,
    AnnihilationOperator,
    NumberOperator,
    ScalarOperator,
    ScaledOperator,
    OperatorSum,
    OperatorProduct
)

# Import Jordan-Schwinger mapping
from .jordan_schwinger import (
    JordanSchwingerOperator,
    jordan_schwinger_map,
    ObservableTensor
)

# Import operator arithmetic and ObservableTensor
from .operator_arithmetic import (
    creation_op, annihilation_op, number_op,
    a_dag, a, n,
    total_photon_number, position_op, momentum_op,
    coherent_displacement, squeeze_operator, beamsplitter_operator,
    commutator, anticommutator
)

__all__ = [
    # QOPTCRAFT base
    'creation_fock', 'annihilation_fock', 'adjoint_evol', 
    'qft', 'qft_inv', 'random_quasiunitary',
    
    # Core operator classes
    'QuantumOperator', 'CreationOperator', 'AnnihilationOperator', 
    'NumberOperator', 'ScalarOperator', 'ScaledOperator', 
    'OperatorSum', 'OperatorProduct',
    
    # Jordan-Schwinger mapping
    'JordanSchwingerOperator', 'jordan_schwinger_map', 'ObservableTensor',
    
    # Basic operators
    'creation_op', 'annihilation_op', 'number_op', 'a_dag', 'a', 'n',
    
    # Composite operators
    'total_photon_number', 'position_op', 'momentum_op',
    'coherent_displacement', 'squeeze_operator', 'beamsplitter_operator',
    
    # Utilities
    'commutator', 'anticommutator',
]